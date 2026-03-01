import torch
import os
import sys
import pickle  # For saving and loading Python objects

# Try importing safetensors, fallback if failed
try:
    from safetensors.torch import load_file as safe_load_file
    use_safetensors = True
except ImportError:
    print("Warning: safetensors library not installed. If checkpoint is .safetensors format, it will not be loaded.")
    print("You can install it via 'pip install safetensors'.")
    use_safetensors = False

# Null space ratio - percentage of smallest singular values to keep as null space
NULL_SPACE_RATIO = 0.80  # Last 80% of singular values

def compute_and_save_lora_null_space_bases(lora_checkpoint_path, output_basis_file):
    """
    Load LoRA weight files, compute (LoRA_B @ LoRA_A) matrix for each layer q,k,v,o_proj, ffn_proj,
    perform SVD, and compute null space basis matrix U based on the last 5% of singular values, then save these basis matrices.
    
    Compared to saving the complete projection matrix P=U@U.T, saving only U can significantly reduce memory usage.

    Args:
        lora_checkpoint_path (str): Path to LoRA weight file or folder.
        output_basis_file (str): Path to .pkl file to save all null space basis matrices.
    """
    
    # 1. Determine the actual LoRA weight file path
    lora_file_to_load = None
    if os.path.isdir(lora_checkpoint_path):
        potential_files = [
            os.path.join(lora_checkpoint_path, "adapter_model.bin"),
            os.path.join(lora_checkpoint_path, "adapter_model.safetensors"),
            os.path.join(lora_checkpoint_path, "pytorch_model.bin"),
            os.path.join(lora_checkpoint_path, "model.safetensors"),
        ]
        for p_file in potential_files:
            if os.path.exists(p_file):
                lora_file_to_load = p_file
                break
        
        if lora_file_to_load is None:
            print(f"Error: LoRA path '{lora_checkpoint_path}' is a folder, but no common LoRA weight files found.")
            print("Please confirm the actual name of the LoRA weight file and ensure it's in the specified folder, or provide the complete file path directly.")
            sys.exit(1)

    elif os.path.isfile(lora_checkpoint_path):
        lora_file_to_load = lora_checkpoint_path
    else:
        print(f"Error: LoRA path '{lora_checkpoint_path}' does not exist or is not a valid file/folder.")
        sys.exit(1)

    print(f"Loading LoRA weight file: {lora_file_to_load}")

    # Load state_dict
    state_dict = {}
    file_extension = os.path.splitext(lora_file_to_load)[1].lower()

    try:
        if file_extension == ".safetensors" and use_safetensors:
            state_dict = safe_load_file(lora_file_to_load)
        elif file_extension in [".bin", ".pt", ".pth"]:
            state_dict = torch.load(lora_file_to_load, map_location='cpu')
        else:
            print(f"Error: Unsupported file format '{file_extension}'. Please provide .bin, .pt, .pth or .safetensors files.")
            sys.exit(1)

    except Exception as e:
        print(f"Error occurred while loading weight file: {e}")
        print("Please ensure the file is in valid PyTorch or safetensors format.")
        sys.exit(1)

    print(f"Successfully loaded {len(state_dict)} parameters.")

    # 2. Iterate through all LoRA parameters and compute null space basis matrices
    # Store all computed null space basis matrices (U matrices)
    null_space_bases = {} 
    
    # Record memory savings
    total_projection_matrix_size = 0
    total_basis_matrix_size = 0

    # Common LoRA layer naming patterns
    lora_a_keys = [key for key in state_dict.keys() if ".lora_A.weight" in key]

    if not lora_a_keys:
        print("No LoRA A matrices found in the LoRA weight file.")
        print("Here are all loaded keys, you can search manually:")
        for key in state_dict.keys():
            print(f"- {key}")
        return

    print(f"\nFound {len(lora_a_keys)} LoRA A matrices, starting processing...")

    for lora_a_key in lora_a_keys:
        # Construct corresponding LoRA B matrix key name
        lora_b_key = lora_a_key.replace(".lora_A.weight", ".lora_B.weight")

        if lora_b_key not in state_dict:
            print(f"Warning: Found LoRA A matrix '{lora_a_key}', but corresponding LoRA B matrix '{lora_b_key}' not found, skipping.")
            continue
        
        lora_a_weight = state_dict[lora_a_key].float().cpu()
        lora_b_weight = state_dict[lora_b_key].float().cpu()

        print(f"\nProcessing module: {lora_a_key.rsplit('.lora_A.weight', 1)[0]}")
        print(f"  LoRA A shape: {lora_a_weight.shape}")
        print(f"  LoRA B shape: {lora_b_weight.shape}")

        # 3. Compute LoRA matrix product (LoRA_B @ LoRA_A)
        try:
            lora_update_matrix = torch.matmul(lora_b_weight, lora_a_weight)
            print(f"  Computed LoRA update matrix (LoRA_B @ LoRA_A), shape: {lora_update_matrix.shape}")
        except RuntimeError as e:
            print(f"  Error: Cannot directly perform matrix multiplication on LoRA_B and LoRA_A. Error message: {e}")
            print("  This may mean the LoRA matrix arrangement is different from expected, trying to transpose one of them.")
            try:
                # Try transposing LoRA_A
                lora_update_matrix = torch.matmul(lora_b_weight, lora_a_weight.T)
                print(f"  Successfully computed LoRA update matrix (LoRA_B @ LoRA_A.T), shape: {lora_update_matrix.shape}")
            except RuntimeError as e_t:
                print(f"  Still failed after trying to transpose LoRA_A. Error message: {e_t}")
                print("  Skipping projection computation for this module.")
                continue

        # 4. Perform Singular Value Decomposition (SVD)
        U, S, Vh = torch.linalg.svd(lora_update_matrix, full_matrices=False)
        
        # Print intermediate matrix rank (based on LoRA theoretical rank)
        tolerance = torch.finfo(S.dtype).eps * max(lora_update_matrix.shape)
        numerical_rank = (S > tolerance).sum().item()
        
        inferred_lora_rank = lora_a_weight.shape[0]  # Usually the first dimension of lora_A
        print(f"  LoRA configured rank (r): {inferred_lora_rank}")
        print(f"  Numerical rank of LoRA update matrix: {numerical_rank}")
        print(f"  Largest top 10 singular values: {[f'{val:.4e}' for val in S[:10].tolist()]}")
        
        # 5. Compute null space basis matrix (based on last 5% of singular values)
        total_singular_values = S.numel()
        null_space_count = max(1, int(total_singular_values * NULL_SPACE_RATIO))  # Keep at least 1
        null_space_start_idx = total_singular_values - null_space_count
        
        print(f"  Total singular values: {total_singular_values}")
        print(f"  Null space ratio: {NULL_SPACE_RATIO*100:.1f}%")
        print(f"  Null space singular value count: {null_space_count}")
        print(f"  Null space singular value range: [{null_space_start_idx}:{total_singular_values}]")
        
        if null_space_count >= total_singular_values:
            print(f"  Warning: Null space contains all singular values, this may be unreasonable. Skipping projection computation for this module.")
            continue
            
        # Extract singular vectors corresponding to null space (U column vectors)
        # Take the last null_space_count singular vectors as null space basis
        U_null_space_basis = U[:, null_space_start_idx:]
        
        # Print singular value range corresponding to null space
        null_space_singular_values = S[null_space_start_idx:]
        print(f"  Null space singular value range: [{null_space_singular_values.min().item():.4e}, {null_space_singular_values.max().item():.4e}]")
        print(f"  Main space singular value range: [{S[null_space_start_idx-1].item():.4e}, {S[0].item():.4e}]")
        
        print(f"  Null space basis matrix shape: {U_null_space_basis.shape}")
        
        # Calculate memory savings
        projection_matrix_size = lora_update_matrix.shape[0] * lora_update_matrix.shape[0] * 4  # float32
        basis_matrix_size = U_null_space_basis.shape[0] * U_null_space_basis.shape[1] * 4  # float32
        total_projection_matrix_size += projection_matrix_size
        total_basis_matrix_size += basis_matrix_size
        
        memory_saved_mb = (projection_matrix_size - basis_matrix_size) / (1024 * 1024)
        print(f"  Memory saved: {memory_saved_mb:.2f} MB (basis matrix vs complete projection matrix)")
        
        # Store null space basis matrix
        module_base_name = lora_a_key.rsplit(".lora_A.weight", 1)[0]
        null_space_bases[module_base_name] = U_null_space_basis

    # 6. Save all null space basis matrices to file
    output_dir = os.path.dirname(output_basis_file)
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    
    with open(output_basis_file, 'wb') as f:
        pickle.dump(null_space_bases, f)
    
    # Calculate total memory savings
    total_memory_saved_mb = (total_projection_matrix_size - total_basis_matrix_size) / (1024 * 1024)
    memory_ratio = total_basis_matrix_size / total_projection_matrix_size if total_projection_matrix_size > 0 else 0
    
    print(f"\nAll {len(null_space_bases)} null space basis matrices saved to: {output_basis_file}")
    print(f"Null space definition: Based on last {NULL_SPACE_RATIO*100:.1f}% of singular values")
    print(f"Total memory saved: {total_memory_saved_mb:.2f} MB")
    print(f"Total basis matrix size: {total_basis_matrix_size/(1024*1024):.2f} MB")
    print(f"Total complete projection matrix size: {total_projection_matrix_size/(1024*1024):.2f} MB")
    print(f"Memory usage ratio: {memory_ratio:.1%}")

def apply_null_space_projection(vector, null_space_basis):
    """
    Project a vector using null space basis matrix.
    
    Args:
        vector: Vector to project, shape (d,) or (d, n)
        null_space_basis: Null space basis matrix U, shape (d, r)
    
    Returns:
        Projected vector, equivalent to (U @ U.T) @ vector
    """
    # Projection formula: P @ v = U @ (U.T @ v)
    # This computation is more memory efficient than first computing P = U @ U.T then multiplying by v
    return torch.matmul(null_space_basis, torch.matmul(null_space_basis.T, vector))

# --- Main program entry point ---
if __name__ == "__main__":
    lora_checkpoint_path = "your_lora_checkpoint_path_here" 
    output_basis_file = "./llama-safety_new_nullspace_bases_1%.pkl" 

    # Before running, please ensure you have installed the following libraries:
    # pip install torch safetensors
    
    compute_and_save_lora_null_space_bases(lora_checkpoint_path, output_basis_file)

    # Example: How to load and use these null space basis matrices
    print(f"\n--- Example: How to load and use null space basis matrices ---")
    if os.path.exists(output_basis_file):
        try:
            with open(output_basis_file, 'rb') as f:
                loaded_bases = pickle.load(f)
            
            print(f"Successfully loaded {len(loaded_bases)} null space basis matrices from '{output_basis_file}'.")
            
            # Print one example
            if loaded_bases:
                first_key = list(loaded_bases.keys())[0]
                first_basis = loaded_bases[first_key]
                print(f"Example null space basis matrix '{first_key}' shape: {first_basis.shape}")
                print(f"Example null space basis matrix '{first_key}' partial content:\n{first_basis[:5, :5]}")
                
                # Demonstrate how to use
                print(f"\nUsage example:")
                print("# Assume you have a gradient or update vector delta")
                print("# delta = torch.randn(first_basis.shape[0])")
                print("# Project it to null space:")
                print("# projected_delta = apply_null_space_projection(delta, first_basis)")
                print("#")
                print("# Or in batch case:")
                print("# batch_deltas = torch.randn(first_basis.shape[0], batch_size)")
                print("# projected_batch = apply_null_space_projection(batch_deltas, first_basis)")
            else:
                print("Loaded null space basis matrix dictionary is empty.")

            print(f"\nYou can use these basis matrices in your DPO process, after computing update matrices (or their gradients):")
            print("Example: projected_delta = apply_null_space_projection(delta, loaded_bases['your_module_path'])")
            print(f"Note: These null space basis matrices are constructed based on the last {NULL_SPACE_RATIO*100:.1f}% of singular values.")
            print("The projection operation is equivalent to the original P @ delta, but more memory efficient.")

        except Exception as e:
            print(f"Error occurred while loading or using saved basis matrix file: {e}")