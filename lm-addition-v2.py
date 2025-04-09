#%% Imports

import math
import os
import random
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, TypeAlias
from jaxtyping import Float
from torch.cuda import OutOfMemoryError  # Add this import for OOM error handling
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import einops
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from IPython.display import IFrame, clear_output, display
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, HookedTransformer
from sklearn.linear_model import LinearRegression
t.set_grad_enabled(False)

# %% Helper functions
# Outputting tensor shapes and printing GPU memory
import inspect
def s(tensor):
    """
    Simple helper function to print the shape of a tensor.
    
    Args:
        tensor: A PyTorch tensor or any object with a .shape attribute
    
    Example:
        attnout = torch.randn(32, 768)
        s(attnout)  # Output: shape of attnout is torch.Size([32, 768])
    """
    # Get the name of the variable from the caller's frame
    frame = inspect.currentframe().f_back
    calling_line = inspect.getframeinfo(frame).code_context[0].strip()
    # Extract variable name from the function call
    # This looks for s(variable_name) pattern
    import re
    match = re.search(r's\((.*?)\)', calling_line)
    if match:
        var_name = match.group(1).strip()
    else:
        var_name = "tensor"
        
    if hasattr(tensor, 'shape'):
        print(f"Shape of [{var_name}]: {tensor.shape}")
    else:
        print(f"{var_name} has no shape attribute. Type: {type(tensor)}")
        
def print_gpu_memory(start_str: str = ""):
    if t.cuda.is_available():
        print(start_str)
        for i in range(t.cuda.device_count()):
            total = t.cuda.get_device_properties(i).total_memory / 1024**3  # Convert to GB
            reserved = t.cuda.memory_reserved(i) / 1024**3
            allocated = t.cuda.memory_allocated(i) / 1024**3
            print(
                f"GPU {i}:",
                f"reserved/allocated/total: {reserved:.2f}/{allocated:.2f}/{total:.2f}",
            )

def clear_gpu_memory(device_id: int = None):
    if t.cuda.is_available():
        t.cuda.empty_cache()
        # Delete models explicitly
        try:
            del gemma
            del gemma_saes
            del gpt
        except:
            pass
        # Force garbage collection
        import gc
        a = gc.collect()
        print(a)
        print("Cleared GPU memory and deleted models")
# %% Loading gemma-2-2b
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print_gpu_memory("before loading gemma")

# Release memory on cuda:0
if t.cuda.is_available():
    with t.cuda.device("cuda:0"):
        t.cuda.empty_cache()
        t.cuda.reset_peak_memory_stats()

# Load models with explicit device placement
gemma = HookedTransformer.from_pretrained("gemma-2-2b", device="cuda:0", dtype=t.bfloat16)
print_gpu_memory("after loading gemma")

#  Calculate size of gemma model parameters
total_params = sum(p.numel() * p.element_size() for p in gemma.parameters())
print(f"Gemma model size: {total_params / 1024**3:.2f} GB")

# Print GPU memory usage after loading gemma
print_gpu_memory("after loading gemma")

# %% Loading 6B model
gpt = HookedTransformer.from_pretrained_no_processing("gpt-j-6b", device="cuda:1", dtype=t.bfloat16)
print_gpu_memory("after loading gpt")
# Briefly test. The device for gemma-2-2b is cuda:0, and the device for gpt-j-6b is cuda:1
#gpt.to("cuda:0") # for some reason this is necessary even if I have specified the device when loading the model...

print(f"gemma device: {gemma.cfg.device}")
print(f"gpt device: {gpt.cfg.device}")
print_gpu_memory("after loading gpt")
#%% moving gpt to cuda1
gpt.to("cuda:1")

# %% Prompt generating function
from collections import namedtuple

AnsConfig = namedtuple("AnsConfig", ["a", "b", "operation", "ans"])

def prompt_generator(
    n_range: int = 100,
    op: list[str] = ["plus"],
    n_batch: int = 100,
    use_grid_search: bool = False,
    write_to_file: bool = True,
    file_path: str = "addition_prompts.txt",
    with_instructions: Literal["none", "instruct", "example"] = "none",
    instr_str: str = "Output ONLY a number",
    with_symbols: bool = False,
) -> tuple[list[str], list[AnsConfig]]:
    """Generates a list of arithmetic questions and their answers.
    """
   
    
    if use_grid_search:
        # Create meshgrid for systematic testing of number combinations
        a_grid = t.arange(0, n_range)
        b_grid = t.arange(0, n_range)
        a, b = t.meshgrid(a_grid, b_grid)
        # Flatten the grids to get all combinations
        a = a.flatten()
        b = b.flatten()
        n_batch = len(a) # Update batch size to total number of combinations
    else:
        a = t.randint(0, n_range, (n_batch,))
        b = t.randint(0, n_range, (n_batch,))
    
    a_instr = t.randint(0, n_range, (n_batch,))
    b_instr = t.randint(0, n_range, (n_batch,))
    
    ans_list = []
    q_list = []
    
    
    with open(file_path, "w") as f:
        for i in range(n_batch):
            
            
            operation = random.choice(op)

            if with_symbols:
                equal_str = "="
                if operation == "plus":
                    operation_str = "+"
                elif operation == "minus":
                    operation_str = "-"
                elif operation == "times":
                    operation_str = "*"
                elif operation == "divided by":
                    operation_str = "/"
                else:
                    raise ValueError("Operation type not recognized")
            else:
                equal_str = "is"

            # log the correct answer
            if operation == "plus":
                answer = a[i] + b[i]
                inst_answer = a_instr[i] + b_instr[i]
            elif operation == "minus":
                answer = a[i] - b[i]
                inst_answer = a_instr[i] - b_instr[i]
            elif operation == "times":
                answer = a[i] * b[i]
                inst_answer = a_instr[i] * b_instr[i]
            # elif operation == "divided by":
            #     answer = a[i] / b[i]
            
            if with_instructions == "none":
                q_list.append(
                    f"{a[i].item()} {operation_str if with_symbols else operation} {b[i].item()} {equal_str}"
                )
            elif with_instructions == "instruct":
                q_list.append(
                    f"{instr_str}. {a[i].item()} {operation_str if with_symbols else operation} {b[i].item()} {equal_str}"
                )
            elif with_instructions == "example":
                q_list.append(
                    f"{a_instr[i].item()} {operation_str if with_symbols else operation} {b_instr[i].item()} {equal_str} {inst_answer.item()}, {a[i].item()} {operation_str if with_symbols else operation} {b[i].item()} {equal_str}"
                )
            else:
                raise ValueError("Instruction type not recognized")

            if write_to_file:
                f.write(q_list[-1] + "\n")
            
            ans_list.append(
                AnsConfig(
                    a=a[i].item(),
                    b=b[i].item(),
                    operation=operation,
                    ans=answer.item()
                )
            )
    
    return q_list, ans_list
# %% Test the function
# q_list, ans_list = prompt_generator(
#     n_range=10,
#     op=["plus"],
#     n_batch=10,
#     use_grid_search=True,
#     with_instructions="instruct",
#     instr_str="Output ONLY a number",
#     with_symbols=True,
# )
# print(q_list)
# print(ans_list)


# %% Define prompt configurations for both models
@dataclass
class PromptModelConfig:
    model_name: str = "gpt-j-6b"
    model: HookedTransformer = gpt
    n_range: int = 25
    n_batch: int = 100
    use_grid_search: bool = True 
    with_instructions: Literal["none", "instruct", "example"] = "instruct"
    instr_str: str = "Output ONLY a number"
    with_symbols: bool = False
    max_new_tokens: int = 1
    
cfgs = [
    PromptModelConfig(
        model_name="gpt-j-6b",
        model=gpt,
        n_range=25,
        use_grid_search=True,
        with_instructions="instruct",
        instr_str="Output ONLY a number",
        max_new_tokens=1,
        with_symbols=True,
    ),
    PromptModelConfig(
        model_name="gemma-2-2b",
        model=gemma,
        n_range=25,
        use_grid_search=True,
        with_instructions="instruct",
        instr_str="Output ONLY a number",
        max_new_tokens=4,
        with_symbols=True
    ),
]
#%% Single run test
def single_run_test(model, prompt, solution, max_new_tokens=4):
    question_tokens = model.to_tokens(prompt, padding_side="left")
    answer_tokens = cfg.model.generate(
        question_tokens,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    # process the output
    #print(model.to_string(answer_tokens))
    ans = answer_tokens[:,-max_new_tokens:].tolist()
    ans = cfg.model.to_string(ans)
    #print(ans)
    return solution in ans
#%% test accuracy with prompt

# gpt.to("cpu").cfg.device
# prompt = "Output ONLY a number, 1 + 4 ="
# gemma.to("cuda:0")
# gemma.generate(prompt, max_new_tokens=4, do_sample=False)
# print_gpu_memory()
# gemma.to("cpu")
#%% Memory management
# gemma.to("cpu")
# gpt.to("cpu")
# print(gemma.cfg.device)
# print(gpt.cfg.device)
# gemma.to("cpu")
# gpt.to("cpu")

#%%
def compute_batch(
    question_list_batch: list[str],
    answer_list_batch: list[AnsConfig],
    model: HookedTransformer,
) -> t.Tensor:
    """
    Compute the accuracy of the model on a batch of questions and answers.
    """
    question_tokens_batch = model.to_tokens(question_list_batch).to(model.cfg.device)
    answer_tokens_batch = model.generate(
        question_tokens_batch,
        max_new_tokens=model.cfg.max_new_tokens,
        do_sample=False,
    ).to(model.cfg.device)


try:
    acc_list = []
    for cfg in cfgs:
        cfg.model.to("cuda:0")
        q_list, ans_list = prompt_generator(
            n_range=cfg.n_range,
            op=["plus"],
            n_batch=cfg.n_batch,
            use_grid_search=cfg.use_grid_search,
            with_instructions=cfg.with_instructions,
            instr_str=cfg.instr_str,
            with_symbols=cfg.with_symbols,
        )
        
        # Get the model's output
        question_tokens = cfg.model.to_tokens(q_list, padding_side="left").to(
            cfg.model.cfg.device
        )
        answer_tokens = cfg.model.generate(
            question_tokens,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
        ).to(cfg.model.cfg.device)
        print(cfg.model.cfg.device)
        # process the output
        print(cfg.model.to_string(answer_tokens))
        ans = [
            answer_tokens[i, -cfg.max_new_tokens:].tolist()
            for i in range(len(answer_tokens))
        ]
        ans = cfg.model.to_string(ans)
        
        correct = [
            str(ans_list[i].ans) in ans[i]
            for i in range(len(ans))
        ]
        correct = t.tensor(correct, device=cfg.model.cfg.device)
        acc_log = correct.reshape(cfg.n_range, cfg.n_range)
        acc_list.append(acc_log)
        # Calculate accuracy
        accuracy = correct.float().mean()
        print(f"{cfg.model_name} accuracy: {accuracy.item():2%}")
        cfg.model.to("cpu")
except OutOfMemoryError:
    print_gpu_memory()
    clear_gpu_memory()
    raise


#%% Write a function to get the full colormap by decomposing into minibatches

def get_acc(
    cfg: PromptModelConfig,
    n_range: int = 100,
    minibatch_size: int = 250,
) -> list[t.Tensor]:
    """
    Get the accuracy of the model on the full range of numbers.
    """
    q_list, ans_list = prompt_generator(
        n_range=n_range,
        op=["plus"],
        use_grid_search=cfg.use_grid_search,
        with_instructions=cfg.with_instructions,
        instr_str=cfg.instr_str,
        with_symbols=cfg.with_symbols,
    )
    acc_list = []
    n_batches = len(q_list) // minibatch_size
    for i in tqdm(range(n_batches), desc="Getting accuracy"):
        t.cuda.empty_cache()
        q_list_batch = q_list[i*minibatch_size:(i+1)*minibatch_size]
        ans_list_batch = ans_list[i*minibatch_size:(i+1)*minibatch_size]
        # Get the model's output
        question_tokens_batch = cfg.model.to_tokens(q_list_batch, padding_side="left").to(
            cfg.model.cfg.device
        )
        answer_tokens_batch = cfg.model.generate(
            question_tokens_batch,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
        ).to(cfg.model.cfg.device)
        #.to(cfg.model.cfg.device)
        # process the output
        ans_batch = [
            answer_tokens_batch[i, -cfg.max_new_tokens:].tolist()
            for i in range(len(answer_tokens_batch))
        ]
        ans_batch = cfg.model.to_string(ans_batch)
        
        correct_batch = [
            str(ans_list_batch[i].ans) in ans_batch[i]
            for i in range(len(ans_batch))
        ]
        correct_batch = t.tensor(correct_batch, device="cpu")
        acc_list.append(correct_batch)
        # Calculate accuracy
        accuracy = correct_batch.float().mean()
        # print(f"{cfg.model_name} accuracy: {accuracy.item():2%}")
        # Update tqdm description with current accuracy
        tqdm.write(f"Batch {i+1}/{n_batches}, Accuracy: {accuracy.item():2%}")

    return q_list, ans_list, acc_list

#%% Processing accuracy
import matplotlib.pyplot as plt
# cfgs[1].model.to("cpu")
# cfgs[0].model.to("cpu")
clear_gpu_memory()
@dataclass

class ArithmeticAccuracy:
    accuracy: float
    acc_list: t.Tensor
    q_list: list[str]
    cfg: PromptModelConfig

arithm_data = []
for cfg in cfgs:
    #Memory management
    clear_gpu_memory()
    cfg.model.to("cuda:0")
    q_list, ans_list, acc_list = get_acc(cfg)
    acc_list = t.cat(acc_list)
    arithm_data.append(ArithmeticAccuracy(
        accuracy=acc_list.float().mean(),
        acc_list=acc_list,
        q_list=q_list,
        cfg=cfg,
    ))
    tot_acc = acc_list.float().mean()
    acc_list = acc_list.reshape(100, 100)
    acc_list = acc_list.float().cpu().numpy()
    plt.figure()
    plt.imshow(acc_list, cmap=plt.cm.RdBu_r, vmin=0, vmax=1)
    plt.colorbar()
    plt.set_cmap('Blues_r')
    plt.title(f"{cfg.model_name} accuracy: {tot_acc:.2%}")
    plt.show()
    #Memory management
    # cfg.model.to("cpu")
#%% Colormap for accuracy

# Create colormaps using plotly
# for i, acc_matrix in enumerate(acc_list):
#     # Convert accuracy matrix to numpy for plotting
#     acc_np = acc_matrix.float().cpu().numpy()
    
#     # Create heatmap
#     fig = go.Figure(data=go.Heatmap(
#         z=acc_np,
#         colorscale=[[0, 'white'], [1, 'skyblue']],  # Skyblue to white colorscale
#         zmin=0,
#         zmax=1,
#         colorbar=dict(
#             ticktext=["Not Correct", "Correct"],
#             tickvals=[0, 1]
#         )
#     ))
    
#     # Update layout
#     fig.update_layout(
#         title=f'Accuracy Heatmap for {cfgs[i].model_name}',
#         xaxis_title='First Number',
#         yaxis_title='Second Number',
#         width=430,
#         height=400
#     )
#     fig.show()

#%% test
#print(gpt.embed.W_E.shape)

# %% get the embeddings for the first 100 numbers of 6b

t.cuda.empty_cache()
t.cuda.reset_peak_memory_stats() # this can release memory that are currently allocated but not used in gpus
gpt.cuda()
gpt.to("cuda:1")
print_gpu_memory("before getting embeddings")
numbers = [str(i) for i in range(360)]
tokens = gpt.to_tokens(numbers, prepend_bos=False)
stop_layer = 1 # we are only interested in the representations, i.e. the embeddings. We treat the first layer as an extension of the embedding layer.

_, cache = gpt.run_with_cache(
    tokens,
    stop_at_layer=stop_layer,
    names_filter=[
        f"blocks.{i}.hook_resid_post"
        for i in range(stop_layer)
    ]
)

acts = cache["blocks.0.hook_resid_post"].squeeze(1)


# %% Do PCA
# q = 2
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

trunc = 100

def plot_pca_projection(trunc, acts, q=1, niter=200, msg="", plot_line=True):
    # Perform PCA
    U, S, V = t.pca_lowrank(acts.to(t.float32)[:trunc], q=q, niter=niter)
    s(U)
    s(S)
    s(V)
    
    # Perform exact PCA from t.linalg.svd
    # U, S, Vh = t.linalg.svd(acts.to(t.float32)[:trunc], full_matrices=False)
    # V = Vh.T
    # # S is already a vector of singular values
    # S = S[:q]
    # V = V[:, :q]
    # U = U[:, :q]
    
    # s(U)
    # s(S)
    # s(V)

    principle_dir = einops.einsum(
        V, S, "d_model d_comp, d_comp -> d_model d_comp"
    ).sum(1)

    principle_dir = principle_dir / principle_dir.norm()

    # Project activations onto principal direction
    act_proj = einops.einsum(
        acts.to(t.float32), principle_dir, "batch d_model, d_model -> batch"
    )
    act_proj = (act_proj - act_proj.mean()).detach().cpu().numpy()

    # Create scatter plot
    if plot_line:
        plt.figure()
        plt.scatter(t.arange(trunc), act_proj[:trunc], s=8)
        plt.xlabel('Number')
        plt.ylabel('PC1 Value')
        plt.title(msg)
        # plt.gca().set_aspect(1)  # Sets aspect ratio to 1:2

        # Fit and plot regression line
        x = t.arange(trunc).reshape(-1, 1)  # Reshape for sklearn
        y = act_proj[:trunc]

        reg = LinearRegression().fit(x, y)
        y_pred = reg.predict(x)

        r2 = r2_score(y, y_pred)

        plt.plot(x, y_pred, 'r--',
                label=f'Fitted line: {reg.coef_[0]:.2f}x + {reg.intercept_:.2f}\nR² = {r2:.3f}')
        plt.legend()
    
    return act_proj

plot_pca_projection(trunc, acts, msg="6B")
#%% FFT
from scipy.signal import find_peaks
fraction = 0.5
yrange = 2000
xrange = 360
def plot_fft_analysis(acts, fraction=0.5, yrange=2000, xrange=360, alpha=1):
    fft_acts = t.fft.fft(acts.to(t.float32), dim=0)
    fft_acts_norm = fft_acts.norm(dim=1)

    plt.plot(fft_acts_norm.cpu().numpy())
    plt.ylim(0, yrange)
    plt.xlim(0, xrange*fraction)
    plt.xticks(plt.xticks()[0], [f'{x/xrange:.2f}' for x in plt.xticks()[0]])

    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('FFT of Number Embeddings')

    # Find peaks in the FFT magnitude spectrum
    

    # Convert to numpy array and find peaks
    fft_magnitudes = fft_acts_norm.cpu().numpy()
    peaks, _ = find_peaks(fft_magnitudes, height=500)  # Adjust height threshold as needed

    # Print peak information
    print("\nPeak frequencies:")
    for peak in peaks:
        print(f"Frequency: {peak/xrange:.2f}, Magnitude: {fft_magnitudes[peak]:.1f}")

    # Add peak markers to plot, filtering out relatively insignificant peaks
    peak_magnitudes = fft_magnitudes[peaks]
    magnitude_threshold = peak_magnitudes.mean() + alpha * peak_magnitudes.std()  # Keep peaks above 1 std dev from mean
    significant_peaks = peaks[peak_magnitudes > magnitude_threshold]

    plt.plot(significant_peaks, fft_magnitudes[significant_peaks], ".", color='red', markersize=10, label='Major Peaks')
    plt.legend()
    # Add text labels for significant peaks
    for peak in significant_peaks:
        freq = peak/xrange
        magnitude = fft_magnitudes[peak]
        plt.annotate(f'f={freq:.2f}\nm={magnitude:.0f}', 
                    xy=(peak, magnitude),
                    xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
                    arrowprops=dict(arrowstyle='->'))

plot_fft_analysis(acts, fraction=0.5, yrange=2000, xrange=360, alpha=-1)
gpt.to("cpu")
#%% Do the same for gemma-2-2b

t.cuda.empty_cache()
t.cuda.reset_peak_memory_stats()
gemma.cuda()
print_gpu_memory("before getting embeddings")

numbers = [str(i) for i in range(10)]
tokens = gemma.to_tokens(numbers, prepend_bos=False)
stop_layer = 1 # we are only interested in the representations, i.e. the embeddings. We treat the first layer as an extension of the embedding layer.

_, cache_gemma = gemma.run_with_cache(
    tokens,
    stop_at_layer=stop_layer,
    names_filter=[
        f"blocks.{i}.hook_resid_post"
        for i in range(stop_layer)
    ]
)

acts_gemma = cache_gemma["blocks.0.hook_resid_post"].squeeze(1)

plot_pca_projection(10, acts_gemma, msg="Gemma-2-2b")
#%% Do the same for gemma-2-2b, now with 3-digit numbers
numbers = [f"{i:03d}" for i in range(300)]
tokens = gemma.to_tokens(numbers, prepend_bos=False)

_, cache_gemma_02d = gemma.run_with_cache(
    tokens,
    stop_at_layer=stop_layer,
    names_filter=[
        f"blocks.{i}.hook_resid_post"
        for i in range(stop_layer)
    ]
)

acts_gemma_02d = cache_gemma_02d["blocks.0.hook_resid_post"]
acts_gemma_02d = einops.rearrange(
    acts_gemma_02d,
    "batch d_digit d_model -> batch (d_digit d_model)"
)


plt.figure()
plot_pca_projection(300, acts_gemma_02d, msg="Gemma-2-2b-2d")

plt.figure()
plot_fft_analysis(acts_gemma_02d, fraction=0.5, yrange=6000, xrange=300, alpha=0)
gemma.to("cpu")
#%% 
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

print_gpu_memory()


#%% Finding helix
# clear_gpu_memory()
def get_act_cache(nrange, layer, model, model_name):
    if model_name == "gpt":
        numbers = [f"{i}" for i in range(nrange)]
        tokens = model.to_tokens(numbers, prepend_bos=False)
        _, cache = model.run_with_cache(tokens, stop_at_layer=layer, names_filter=[f"blocks.{i}.hook_resid_post" for i in range(layer)])
        return cache
    elif model_name == "gemma":
        numbers = [f"{i:03d}" for i in range(nrange)]
        tokens = model.to_tokens(numbers, prepend_bos=False)
        _, cache = model.run_with_cache(tokens, stop_at_layer=layer, names_filter=[f"blocks.{i}.hook_resid_post" for i in range(layer)])
        acts = cache[f"blocks.{layer-1}.hook_resid_post"]
        new_cache = {}
        for k,v in cache.items():
            new_cache[k] = einops.rearrange(
                v,
                "batch d_digit d_model -> batch (d_digit d_model)"
            )
        return new_cache


def find_helix(acts, nrange=100, T=[2,5,10,100], plot=False, linear = True):
    t.cuda.empty_cache()
    
    # Create the target basis vectors
    B = []
    for a in range(nrange):
        # Create base tensor with the number
        if linear:
            base = [t.tensor([a])]
        else:
            base = []
        
        # Calculate trig components for each period T
        for period in T:
            angle = t.tensor(2*math.pi*a / period)
            trig_components = t.stack([t.cos(angle), t.sin(angle)])
            base.append(trig_components.flatten())
        
        # Combine components for this number
        B.append(t.cat(base))

    # Stack all number representations into final tensor
    B = t.stack(B)


    # Perform PCA
    U, S, V = t.pca_lowrank(acts.squeeze(1).to(t.float32), q=100, niter=2)
    V_float = V.to(t.float32)
    acts_proj = acts.squeeze(1).to(t.float32) @ V_float

    B_float = B.to(t.float32).to(acts_proj.device)
    # Linear regression
    C_PCA = t.zeros(B_float.shape[1], acts_proj.shape[1]).to(acts_proj.device)
    acts_proj_pred = t.zeros_like(acts_proj)
    for i in range(acts_proj.shape[1]):
        reg = LinearRegression().fit(B_float.cpu().numpy(), acts_proj[:,i].cpu().numpy())
        C_PCA[:,i] = t.from_numpy(reg.coef_)
        acts_proj_pred[:,i] = t.from_numpy(reg.predict(B_float.cpu().numpy()))

        total_ss = ((acts_proj[:,i] - acts_proj[:,i].mean())**2).sum()
        residual_ss = ((acts_proj[:,i] - acts_proj_pred[:,i])**2).sum()
        r_squared = 1 - (residual_ss/total_ss)
        print(f"R-squared score: {r_squared.mean().item():.4f}")

    # B_float = B.to(t.float32).to(acts_proj.device)
    # C_PCA, _, _, _ = t.linalg.lstsq(B_float, acts_proj)
    # acts_proj_pred =  B_float@C_PCA
    total_ss = ((acts_proj - acts_proj.mean(dim=0))**2).sum(dim=0)
    residual_ss = ((acts_proj - acts_proj_pred)**2).sum(dim=0)
    r_squared = 1 - (total_ss/residual_ss)
    print(f"R-squared score_final: {r_squared.mean().item():.4f}")


    
    C = C_PCA@V_float.T
    helix_proj = acts.squeeze(1).to(t.float32) @ C.pinverse()
    acts_pred = B_float @ C

    if plot:
        cols = len(T)
        n_points = len(helix_proj)
        start = 0
        if linear:
            fig = px.scatter(y=helix_proj[:,0].cpu().numpy())
            fig.show()
            start = 1
        # Create subplots with colored points
        fig_combined = make_subplots(rows=1, cols=cols)
        for i, cur_t in enumerate(T):

            fig_combined.add_trace(go.Scatter(x=helix_proj[:,start+i*2].cpu().numpy(), 
                                        y=helix_proj[:,start+i*2+1].cpu().numpy(),
                                        mode='markers',
                                        marker=dict(color=np.arange(n_points) % cur_t)), row=1, col=i+1)
       
        fig_combined.update_layout(height=400, width=1200, showlegend=False)
        fig_combined.show()
    return acts_pred

#%% The helix
#px.line(t.arange(nrange), helix_proj_[:,0].cpu().numpy())
# gpt.to("cuda:0")
acts_gpt_cache = get_act_cache(100, 1, gpt, "gpt")
acts_gpt_pred = find_helix(acts_gpt_cache["blocks.0.hook_resid_post"], nrange=100, T=[2,5,10,100], plot=True, linear=True)
# gpt.to("cpu")
gpt.to("cuda:1")
# T = [2,5,10,100]
# helix_proj = t.randn(100,9).to("cuda:0")
# cols = len(T)
# n_points = len(helix_proj)
# start = 0
# if linear:
#     fig = px.scatter(y=helix_proj[:,0].cpu().numpy())
#     fig.show()
#     start = 1
# Create subplots with colored points
# fig_combined = make_subplots(rows=1, cols=cols)
# for i, cur_t in enumerate(T):

#     fig_combined.add_trace(go.Scatter(x=helix_proj[:,start+i*2].cpu().numpy(), 
#                                 y=helix_proj[:,start+i*2+1].cpu().numpy(),
#                                 mode='markers',
#                                 marker=dict(color=np.arange(n_points) % cur_t)), row=1, col=i+1)

# fig_combined.update_layout(height=400, width=1200, showlegend=False)
# fig_combined.show()

gemma.to("cuda:0")
acts_gemma_cache = get_act_cache(100, 1, gemma, "gemma")
acts_gemma_pred = find_helix(acts_gemma_cache["blocks.0.hook_resid_post"], nrange=100, T=[2, 5, 10], plot=True, linear=True)
# gemma.to("cpu")




#%% Activation patching: Generate 100 pairs of random number sums

batch_size = 100
def generate_patching_instances(
    model: HookedTransformer,
    arithm_data: ArithmeticAccuracy,
    batch_size: int = 100,
    nrange: int = 100,
    seed: int | None = None,
):
    t.cuda.manual_seed(seed)
    answers = []
    prompt_pairs = []
    all_prompts = arithm_data.acc_list.flatten().tolist()
    #print(len(all_prompts))
    correct_indices = [i for i, x in enumerate(all_prompts) if x]
    correct_prompts= []
    corrupt_prompts = []
    select_correct_indices = random.sample(correct_indices, batch_size)
    select_correct = [arithm_data.q_list[i] for i in select_correct_indices]
    for prompt in select_correct:
        a = prompt.split("+")[0].split("Output ONLY a number.")[1].strip()
        b = prompt.split("+")[1].split("=")[0].strip()
        a_ = random.randint(0, nrange)
        while a_ == int(a):
            a_ = random.randint(0, nrange)
        corrupt_prompt = f"Output ONLY a number. {a_} + {b} ="

        correct_prompts.append(prompt)
        corrupt_prompts.append(corrupt_prompt)
        
        answers.append(
            str(int(a) + int(b))
        )
    return correct_prompts, corrupt_prompts, answers

#gemma.to("cuda:0")
correct_prompts_gemma, corrupt_prompts_gemma, answers_gemma = generate_patching_instances(gemma, arithm_data[0], seed=42)

#gemma.to("cpu")
#gpt.to("cuda:0")
correct_prompts_gpt, corrupt_prompts_gpt, answers_gpt = generate_patching_instances(gpt, arithm_data[1], seed=42)
#gpt.to("cpu")

print(correct_prompts_gpt)
print(corrupt_prompts_gpt)
print(answers_gpt)

#%% ACtivation patching: PCA patching


# %% patching the entire layer

t.cuda.empty_cache()
# a_cl, b = correct_list[0]
# a_cor = corrupt_list[0][0]
# lyr = 1
gpt.to("cuda:1")
corrupt_logits = gpt(
    corrupt_prompts_gpt,
    return_type="logits",
)[:, -1, :] # Only take logits for the last position

#%%

#%% Hooks that patch the entire layer
from functools import partial
from transformer_lens.hook_points import HookPoint
def replace_with_clean(
    corrupt_acts: Float[t.Tensor, "batch pos d_model"],
    hook: HookPoint,
    clean_acts: Float[t.Tensor, "batch pos d_model"],
    layer: int | None = None,
    seq_pos: int | None = None,
) -> Float[t.Tensor, "batch pos d_model"]:
    """Replace the corrupt acts with the clean acts"""
    assert corrupt_acts.shape == clean_acts.shape
    corrupt_acts[:, seq_pos, :] = clean_acts[:, seq_pos, :]
    return corrupt_acts


def get_clean_acts(
    model: HookedTransformer,
    correct_prompts: list[str]
) -> ActivationCache:
    _, cache = model.run_with_cache(
        correct_prompts,
        names_filter=[f"blocks.{layer}.hook_resid_post" for layer in range(model.cfg.n_layers)],
    )
    return cache

#%% Hooks that patch the PCA

def get_pca_acts(
    model: HookedTransformer,
    correct_prompts: list[str],
    q: int = 9,
) -> ActivationCache:
    # get the PCA replacement at blocks.0.hook_resid_post
    _, cache = model.run_with_cache(
        correct_prompts,
        stop_at_layer=1,
        names_filter=["blocks.0.hook_resid_post"]
    )
    acts = cache["blocks.0.hook_resid_post"][:, -4, :] # position of the first number
    U, S, V = t.pca_lowrank(acts.to(t.float32), q=q, niter=200)
    recon_acts_from_pc_q = acts.to(t.float32) @ V @ V.T

    # now obtain the Full activation cache 
    # consider workarounds with forward
    hooked_resid_stream_0 = cache["blocks.0.hook_resid_post"]
    hooked_resid_stream_0[:, -4, :] = recon_acts_from_pc_q
    resid_stream_cache = {}
    resid_stream_cache["blocks.0.hook_resid_post"] = hooked_resid_stream_0
    # this should now be (batch, pos, d_model)
    
    layer = 0
    while layer < model.cfg.n_layers:
        resid_stream_next = model.forward(
            resid_stream_cache["blocks.0.hook_resid_post"],
            start_at_layer=layer,
            stop_at_layer=layer+1
        )
        resid_stream_cache[f"blocks.{layer+1}.hook_resid_post"] = resid_stream_next
        layer += 1
    
    return resid_stream_cache

def replace_with_pca(
    corrupt_acts: Float[t.Tensor, "batch pos d_model"],
    hook: HookPoint,
    pca_acts: Float[t.Tensor, "batch pos d_model"],
    layer: int | None = None,
    seq_pos: int | None = None,
) -> Float[t.Tensor, "batch pos d_model"]:
    """Replace the corrupt acts with the PCA acts"""
    if seq_pos is None:
        corrupt_acts = pca_acts
    else:
        corrupt_acts[:, seq_pos, :] = pca_acts[:, seq_pos, :]
        
    return corrupt_acts

#%% Another guess: Use PCA approximation of the residual stream at each layer

def get_pca_acts_v1(
    model: HookedTransformer,
    correct_prompts: list[str],
    q: int = 9,
) -> ActivationCache:
    # get the PCA replacement at blocks.0.hook_resid_post
    _, cache = model.run_with_cache(
        correct_prompts,
        names_filter=[f"blocks.{l}.hook_resid_post" for l in range(model.cfg.n_layers)]
    )
    # replace each layer (-4) position with the PCA approximation
    for l in range(model.cfg.n_layers):
        acts = cache[f"blocks.{l}.hook_resid_post"][:, -4, :]
        U, S, V = t.pca_lowrank(acts.to(t.float32), q=q, niter=200)
        recon_acts_from_pc_q = acts.to(t.float32) @ V @ V.T
        cache[f"blocks.{l}.hook_resid_post"][:, -4, :] = recon_acts_from_pc_q

    return cache

#%% Patch with entire layer
t.cuda.empty_cache()
clean_cache = get_clean_acts(gpt, correct_prompts_gpt)
# Get the tokens corresponding to the correct answers

correct_answer_tokens = []
for answer in answers_gpt:
    # Convert answer string to token IDs
    tokens = gpt.to_tokens(str(answer), prepend_bos=False)
    # Get just the last token (the answer token)
    correct_answer_tokens.append(tokens)


correct_answer_token_indices = t.tensor(correct_answer_tokens).to(corrupt_logits.device)
# this is a list of indices (batch,) labelling the correct answer token index for each prompt
# print("Tokens for correct answers:", correct_answer_tokens)
# print("Token strings:", [gpt.to_string(token) for token in correct_answer_tokens])

# the baseline of corrupted logits
corrupt_logits = gpt(
    corrupt_prompts_gpt,
    return_type="logits",
)[:, -1, :]

# run with hooks to patch the correct activations
patched_logits = []
for lyr in tqdm(range(gpt.cfg.n_layers)):
    patched_logit = gpt.run_with_hooks(
        corrupt_prompts_gpt,
        fwd_hooks=[
            (
                f"blocks.{lyr}.hook_resid_post",
                partial(
                    replace_with_clean,
                    clean_acts=clean_cache[f"blocks.{lyr}.hook_resid_post"],
                    seq_pos=-4 # referring to the first number token
                ),
            )
        ],
        return_type="logits",
        reset_hooks_end=True,
    )[:, -1, :]
    patched_logits.append(patched_logit)

patched_logits_at_layers = t.stack(patched_logits)
# shape is (n_layers, batch, d_vocab)

#%% Patch with PCA

resid_stream_cache_pca = get_pca_acts(gpt, correct_prompts_gpt, q=9)

patched_logits_pca = []
for lyr in tqdm(range(gpt.cfg.n_layers)):
    patched_logit = gpt.run_with_hooks(
        corrupt_prompts_gpt,
        fwd_hooks=[
            (
                f"blocks.{lyr}.hook_resid_post",
                partial(replace_with_pca, pca_acts=resid_stream_cache_pca[f"blocks.{lyr}.hook_resid_post"], seq_pos=-4)
            )
        ],
        return_type="logits",
        reset_hooks_end=True,
    )[:, -1, :]
    patched_logits_pca.append(patched_logit)

patched_logits_pca_at_layers = t.stack(patched_logits_pca)




#%% Patch with PCA v1

resid_stream_cache_pca_v1 = get_pca_acts_v1(gpt, correct_prompts_gpt, q=9)

patched_logits_pca_v1 = []
for lyr in tqdm(range(gpt.cfg.n_layers)):
    patched_logit = gpt.run_with_hooks(
        corrupt_prompts_gpt,
        fwd_hooks=[
            (
                f"blocks.{lyr}.hook_resid_post",
                partial(replace_with_pca, pca_acts=resid_stream_cache_pca_v1[f"blocks.{lyr}.hook_resid_post"], seq_pos=-4)
            )
        ],
        return_type="logits",
        reset_hooks_end=True,
    )[:, -1, :]
    patched_logits_pca_v1.append(patched_logit)

patched_logits_pca_v1_at_layers = t.stack(patched_logits_pca_v1)

#%%
def plot_logit_diff(
    patched_logits: Float[t.Tensor, "n_layers batch d_vocab"],
    corrupt_logits: Float[t.Tensor, "batch d_vocab"],
    correct_answer_token_indices: t.Tensor,
    plot_line: bool = True,
) -> None:
    logit_diff = (
        patched_logits - corrupt_logits.unsqueeze(0).repeat(gpt.cfg.n_layers, 1, 1)
    ).gather(
        dim=-1,
        index=correct_answer_token_indices.unsqueeze(0).unsqueeze(2).expand(gpt.cfg.n_layers, -1, 1)
    )
    if plot_line:
        plt.figure(figsize=(10, 6))
        plt.plot(logit_diff.mean(1).to(t.float32).cpu().numpy())
        plt.xlabel("Layer")
        plt.ylabel("Mean Logit Difference") 
        plt.title("Logit Difference by Layer When Patching Residual Stream")
        plt.grid(True)
        plt.show()

    return logit_diff

# logit_diff_clean = plot_logit_diff(patched_logits_at_layers, corrupt_logits, correct_answer_token_indices, plot_line=True)
logit_diff_pca = plot_logit_diff(patched_logits_pca_at_layers, corrupt_logits, correct_answer_token_indices, plot_line=True)
# logit_diff_pca_v1 = plot_logit_diff(patched_logits_pca_v1_at_layers, corrupt_logits, correct_answer_token_indices, plot_line=True)



