import torch
import numpy as np
from einops import rearrange, repeat
from typing import Optional
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster, Forecast
from toto.data.util.dataset import pad_array, pad_id_mask
from util import get_toto_activations, create_attention_mask_for_layer, apply_style_transfer
class StylizedTotoForecaster(TotoForecaster):
    """
    An enhanced TotoForecaster that supports activation style transfer for
    controllable, counterfactual forecasting.

    This class inherits from the original TotoForecaster and adds a new method,
    `stylized_forecast`, which generates a forecast for a "content" time series
    under the volatility "style" of another time series. The intervention is
    performed within a custom autoregressive generation loop.
    """

    @torch.no_grad()
    def stylized_forecast(
        self,
        content_inputs: MaskedTimeseries,
        style_inputs: MaskedTimeseries,
        intervention_layer_idx: int,
        prediction_length: int,
        num_samples: int = 256,
        samples_per_batch: int = 64,
        use_kv_cache: bool = True,
    ) -> Forecast:
        """
        Generates a forecast for the content_inputs using the style of style_inputs.

        Args:
            content_inputs (MaskedTimeseries): The time series providing the "content" (e.g., trend).
            style_inputs (MaskedTimeseries): The time series providing the "style" (e.g., volatility).
            intervention_layer_idx (int): The transformer layer index to perform the style transfer.
            prediction_length (int): The number of future time steps to predict.
            num_samples (int): The number of sample trajectories to generate.
            samples_per_batch (int): The number of samples to process in a single batch to manage memory.
            use_kv_cache (bool): Whether to use a KV cache for faster autoregressive generation.

        Returns:
            Forecast: A forecast object containing the mean and samples of the stylized prediction.
        """
        # We use super().__init__ implicitly.
        # This method mirrors the structure of the original `forecast` method.

        # --- Data Pre-processing ---
        # We only need to pad the content inputs, as the style is used once.
        content_series = pad_array(content_inputs.series, self.model.patch_embed.stride)
        content_padding_mask = pad_array(content_inputs.padding_mask, self.model.patch_embed.stride)
        content_id_mask = pad_id_mask(content_inputs.id_mask, self.model.patch_embed.stride) if content_inputs.id_mask is not None else None

        # sys.exit(0)
        # Call the new stylized sample generator
        samples = self.generate_stylized_samples(
            content_series=content_series,
            content_padding_mask=content_padding_mask,
            content_id_mask=content_id_mask,
            style_inputs=style_inputs,
            intervention_layer_idx=intervention_layer_idx,
            prediction_length=prediction_length,
            num_samples=num_samples,
            sampling_batch_size=samples_per_batch,
            use_kv_cache=use_kv_cache,
        )
        
        mean = samples.mean(dim=-1)
        return Forecast(mean=mean, samples=samples)


    @torch.no_grad()
    def generate_stylized_samples(
        self,
        content_series: torch.Tensor,
        content_padding_mask: torch.Tensor,
        content_id_mask: Optional[torch.Tensor],
        style_inputs: MaskedTimeseries,
        intervention_layer_idx: int,
        prediction_length: int,
        num_samples: int,
        sampling_batch_size: int,
        use_kv_cache: bool,
    ) -> torch.Tensor:
        """
        Generates sample trajectories for a content time series using the dynamic
        "style" of a style time series via activation intervention.
    
        This method contains a custom autoregressive loop that:
        1. Runs the model forward up to an intervention layer.
        2. Applies style transfer (AdaIN) to the activations.
        3. Completes the forward pass with the modified activations.
        4. Samples the next time step and appends it to the input.
        5. Repeats the process for the entire forecast horizon.
        """
        # --- 1. SETUP & INITIAL ANALYSIS ---
        device = self.model.device
        backbone = self.model # The TotoBackbone is stored in self.model
    
        # Get the "style" activation statistics once, before the main loop begins.
        # This is the "soul" we will inject on every generation step.
        style_dict = {
            'inputs': style_inputs.series,
            'padding_mask': style_inputs.padding_mask,
            'id_mask': style_inputs.id_mask
        }
        # Use our hook-based function to get all layer activations for the style window.
        _, style_full_activations = get_toto_activations(backbone, [style_dict])
        a_style = style_full_activations[intervention_layer_idx].to(device)
        
        # --- 2. PREPARE INPUTS FOR BATCHED SAMPLING ---
        assert num_samples % sampling_batch_size == 0, "num_samples must be divisible by sampling_batch_size"
        
        patch_size = backbone.patch_embed.patch_size
        # Calculate how many generation steps are needed (must be a multiple of patch_size)
        num_patches_to_generate = int(np.ceil(prediction_length / patch_size))
        
        # Store original sequence start index for final trimming
        start_index = content_series.shape[-1]
        
        # Repeat the initial context window for each sample in the batch
        batch_inputs = repeat(
            content_series, "b v s -> (sp b) v s", sp=sampling_batch_size
        )
        # --- 3. THE AUTOREGRESSIVE GENERATION LOOP ---
        num_batches = num_samples // sampling_batch_size
        all_samples_batches = []
    
        for _ in range(num_batches):
            # Clone the initial context for this specific batch of sample paths
            iter_inputs = torch.clone(batch_inputs)
    
            # NOTE: Full KV cache implementation for this custom loop is highly complex.
            # For a workshop paper, disabling it and noting as future work is acceptable.
            # We proceed without KV cache for clarity and correctness.
            
            for _ in range(num_patches_to_generate):
                
                # --- A. PREPARE CURRENT STEP INPUTS ---
                # The model's scaler and patcher need correctly shaped masks
                current_padding_mask = torch.full_like(iter_inputs, True, dtype=torch.bool, device=device)
                current_id_mask = torch.zeros_like(iter_inputs, device=device) if content_id_mask is None else repeat(content_id_mask, "b v s -> (sp b) v s", sp=sampling_batch_size)
                
                # --- B. EXECUTE THE TWO-PART FORWARD PASS ---
                
                # PART 1: Run model from input up to the intervention layer
                scaled_inputs, loc, scale = backbone.scaler(iter_inputs, torch.ones_like(iter_inputs), current_padding_mask)
                embeddings, reduced_id_mask = backbone.patch_embed(scaled_inputs, current_id_mask)
                
                x = embeddings
                # sys.exit(0)
                for i in range(intervention_layer_idx + 1):
                    layer = backbone.transformer.layers[i]
                    # Create the specific mask needed for this layer
                    attention_mask = create_attention_mask_for_layer(layer, reduced_id_mask)
                    x = layer(layer_idx=i, inputs=x, attention_mask=attention_mask)
                a_content = x
                
                # PART 2: Perform Style Transfer and complete the forward pass
                
                
                a_style = a_style.expand_as(a_content)
                stylized_activation = apply_style_transfer(a_content, a_style)

                
                
                x = stylized_activation
                
                for i in range(intervention_layer_idx + 1, len(backbone.transformer.layers) - 1):
                    layer = backbone.transformer.layers[i]
                
                    # Create the specific mask needed for this layer
                    attention_mask = create_attention_mask_for_layer(layer, reduced_id_mask)
                    
                    x = layer(layer_idx=i, inputs=x, attention_mask=attention_mask)


                unembedded = backbone.unembed(x)
                flattened = rearrange(
                    unembedded, "b v s (p e) -> b v (s p) e", 
                    p=patch_size, e=backbone.embed_dim
                )
                # --- C. SAMPLE THE NEXT PATCH ---
                # Get the final distribution and de-normalize it
                base_distr = backbone.output_distribution(flattened)
                distr = super().create_affine_transformed(base_distr, loc, scale)
                
                # Sample only the last patch worth of time steps from the distribution
                new_patch_samples = distr.sample()[:, :, -patch_size:]
                
                # --- D. APPEND AND REPEAT ---
                # Append the newly generated patch to our sequence
                iter_inputs = torch.cat([iter_inputs, new_patch_samples], dim=-1)
    
            # After generating a full trajectory for this batch, save it
            all_samples_batches.append(iter_inputs)
        
        # --- 4. POST-PROCESSING ---
        # Combine the results from all batches
        outputs = torch.cat(all_samples_batches, dim=0)
        
        # Reshape to group by original batch item and then by sample
        # Final shape: [original_batch_size, num_variates, full_seq_len, num_samples]
        unfolded_outputs = rearrange(
            outputs,
            "(samples batch) variates seq_len -> batch variates seq_len samples",
            samples=num_samples,
        )
        
        # Trim the forecast to the exact requested prediction_length
        end_index = start_index + prediction_length
        trimmed_predictions = unfolded_outputs[:, :, start_index:end_index, :]
        
        return trimmed_predictions.detach()
