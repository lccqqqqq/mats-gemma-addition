#%% Imports
import gc
import itertools
import math
import os
import random
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, TypeAlias
from torch.cuda import OutOfMemoryError  # Add this import for OOM error handling
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import einops
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import torch as t
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from IPython.display import HTML, IFrame, clear_output, display
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from sae_lens import (
    SAE,
    ActivationsStore,
    HookedSAETransformer,
    LanguageModelSAERunnerConfig,
    SAEConfig,
    SAETrainingRunner,
    upload_saes_to_huggingface,
)
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
# from sae_vis import SaeVisConfig, SaeVisData, SaeVisLayoutConfig
# from tabulate import tabulate
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name, test_prompt, to_numpy

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

#%% Load models with explicit device placement
gemma = HookedTransformer.from_pretrained("gemma-2-2b", device="cuda:0", dtype=t.bfloat16)
print_gpu_memory("after loading gemma")

# %% Calculate size of gemma model parameters
total_params = sum(p.numel() * p.element_size() for p in gemma.parameters())
print(f"Gemma model size: {total_params / 1024**3:.2f} GB")

# Print GPU memory usage after loading gemma
print_gpu_memory("after loading gemma")

# %% Loading 6B model
gpt = HookedTransformer.from_pretrained_no_processing("gpt-j-6b", device="cuda:1", dtype=t.bfloat16)
print_gpu_memory("after loading gpt")
#%% Briefly test. The device for gemma-2-2b is cuda:0, and the device for gpt-j-6b is cuda:1
gpt.to("cuda:0") # for some reason this is necessary even if I have specified the device when loading the model...

print(f"gemma device: {gemma.cfg.device}")
print(f"gpt device: {gpt.cfg.device}")
print_gpu_memory("after loading gpt")

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
q_list, ans_list = prompt_generator(
    n_range=10,
    op=["plus"],
    n_batch=10,
    use_grid_search=True,
    with_instructions="instruct",
    instr_str="Output ONLY a number",
    with_symbols=True,
)
print(q_list)
print(ans_list)


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
    ),
]

#%% test accuracy with prompt

gpt.to("cuda:1").cfg.device
prompt = "Output ONLY a number, 1 + 4 ="
gemma.generate(prompt, max_new_tokens=4, do_sample=False)
print_gpu_memory()
#%%
print(gemma.cfg.device)
print(gpt.cfg.device)

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
        # cfg.model.to("cuda:0")
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
        # cfg.model.to("cpu")
except OutOfMemoryError:
    print_gpu_memory()
    clear_gpu_memory()
    raise


#%% Write a function to get the full colormap by decomposing into minibatches

def get_acc(
    cfg: PromptModelConfig,
    n_range: int = 100,
    minibatch_size: int = 1000,
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

@dataclass
class ArithmeticAccuracy:
    accuracy: float
    acc_list: t.Tensor
    q_list: list[str]
    cfg: PromptModelConfig

arithm_data = []
for cfg in cfgs:
    q_list, ans_list, acc_list = get_acc(cfg)
    acc_list = t.cat(acc_list)
    arithm_data.append(ArithmeticAccuracy(
        accuracy=acc_list.float().mean(),
        acc_list=acc_list,
        q_list=q_list,
        cfg=cfg
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
print(gpt.embed.W_E.shape)

# %% get the embeddings for the first 100 numbers of 6b

t.cuda.empty_cache()
t.cuda.reset_peak_memory_stats() # this can release memory that are currently allocated but not used in gpus
gpt.cuda()
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

def plot_pca_projection(trunc, acts, q=1, niter=2, msg="", plot_line=True):
    # Perform PCA
    U, S, V = t.pca_lowrank(acts.to(t.float32)[:trunc], q=q, niter=niter)
    s(U)
    s(S) 
    s(V)

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
    pass

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

#%% 
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

print_gpu_memory()


#%% Finding helix
# clear_gpu_memory()
def get_act(nrange, layer, model, model_name):
    if model_name == "gpt":
        numbers = [f"{i}" for i in range(nrange)]
        tokens = model.to_tokens(numbers, prepend_bos=False)
        _, cache = model.run_with_cache(tokens, stop_at_layer=layer, names_filter=[f"blocks.{layer-1}.hook_resid_post"])
        return cache[f"blocks.{layer-1}.hook_resid_post"]
    elif model_name == "gemma":
        numbers = [f"{i:03d}" for i in range(nrange)]
        tokens = model.to_tokens(numbers, prepend_bos=False)
        _, cache = model.run_with_cache(tokens, stop_at_layer=layer, names_filter=[f"blocks.{layer-1}.hook_resid_post"])
        acts = cache[f"blocks.{layer-1}.hook_resid_post"]
        acts = einops.rearrange(
                acts,
                "batch d_digit d_model -> batch (d_digit d_model)"
                )
        return acts

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

    # Linear regression
    B_float = B.to(t.float32).to(acts_proj.device)
    C_PCA, residuals, rank, ss = t.linalg.lstsq(B_float, acts_proj)
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
acts_gpt = get_act(100, 1, gpt, "gpt")
find_helix(acts_gpt, nrange=100, T=[2,5,10,100], plot=True, linear=True)

acts_gemma = get_act(100, 1, gemma, "gemma")
find_helix(acts_gemma, nrange=100, T=[2,5,10,100], plot=True, linear=True)


#%%
acts_gpt_proj = acts_gpt.squeeze(1).to(t.float32) @ V_float
B_float = B.to(t.float32).to("cuda:0")

# Solve the linear regression equation: acts_gpt_proj ≈ B @ W
# Using sklearn's ridge regression with cross-validation
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# Convert to numpy for sklearn
B_np = B_float.cpu().numpy()
acts_np = acts_gpt_proj.cpu().numpy()

# Standardize features
scaler = StandardScaler()
B_scaled = scaler.fit_transform(B_np)

# Fit ridge regression with cross-validation
alphas = [0.1, 1.0, 10.0]  # Regularization parameters to try
reg = RidgeCV(alphas=alphas, cv=5)
reg.fit(B_scaled, acts_np)

# Transform coefficients back to PyTorch tensor and correct for scaling
C_PCA = t.from_numpy(reg.coef_).to(B_float.device).to(t.float32)
residuals = t.from_numpy(
    np.sum((acts_np - reg.predict(B_scaled))**2, axis=0)
).to(B_float.device)

s(residuals)
s(C_PCA)

#%%
# Calculate R-squared score to measure goodness of fit
acts_gpt_proj_pred =  B_float@C_PCA
s(acts_gpt_proj_pred)
total_ss = ((acts_gpt_proj - acts_gpt_proj.mean(dim=0))**2).sum(dim=0)
residual_ss = ((acts_gpt_proj - acts_gpt_proj_pred)**2).sum(dim=0)
r_squared = 1 - (residual_ss/total_ss)
print(residual_ss)
print(total_ss)

print(f"R-squared score: {r_squared.mean().item():.4f}")
print(f"Regression weights shape: {C_PCA.shape}")

#%%
s(B_float)
s(acts_gpt_proj)


#%% Activation patching: Generate 100 pairs of random number sums

batch_size = 100
def generate_patching_instances(
    arith_acc: ArithmeticAccuracy,
    model: HookedTransformer,
    batch_size: int = 100,
    seed: int | None = None,
):
    assert arith_acc.cfg.model == model, "Model mismatch"
    if seed is not None:
        t.manual_seed(seed)
    
    correct_list = []
    corrupt_list = []
    acc_map = arith_acc.acc_list.reshape(100, 100)
    while len(correct_list) < batch_size:
        a = t.randint(0, 100, (1,))
        b = t.randint(0, 100, (1,))
        a1 = t.randint(0, 100, (1,))
        # check that the model can solve both correctly
        if acc_map[a, b] and acc_map[a1, b]:
            correct_list.append((a, b))
            corrupt_list.append((a1, b))
    
    return correct_list, corrupt_list

# TODO: validate that all these instances are correct?
#%% ACtivation patching: PCA patching


#%%
# start with gpt
correct_list, corrupt_list = generate_patching_instances(arithm_data[0], gpt, seed=42)
print(correct_list)
print(corrupt_list)
#%% Generate prompt pairs
nrange = 360
answers = []
prompt_pairs = []
for _ in range(100):
    a = random.randint(0, nrange)
    b = random.randint(0, nrange)
    a_ = random.randint(0, nrange)
    while a_ == a:
        a_ = random.randint(0, nrange)
    prompt_pairs.append(
        (
            f"Output ONLY a number. {a} + {b} =",
            f"Output ONLY a number. {a_} + {b} ="
        )
    )
    answers.append(
        f"{a + b}"
    )
print(prompt_pairs)


# %% patching the entire layer

t.cuda.empty_cache()
a_cl, b = correct_list[0]
a_cor = corrupt_list[0][0]
lyr = 1

clean_logits = gpt(
    arithm_data[0].q_list[a_cl * 100 + b],
    return_type="logits",
)[:, -1, :] # Only take logits for the last position

#%%
print(arithm_data[0].q_list[a_cl * 100 + b])

#%%

def replace_with_clean(
    clean_acts: t.Tensor,
    corrupt_acts: t.Tensor,
):
    return corrupt_acts

def get_clean_acts(
    model: HookedTransformer,
    clean_input: tuple[int, int],
) -> ActivationCache:
    _, cache = model.run_with_cache(
        q_list[clean_input[0] * 100 + clean_input[1]],
        names_filter=[f"blocks.{layer}.hook_resid_post" for layer in range(model.cfg.n_layers)],
    )
    return cache

from functools import partial

corrupt_logits = gpt.run_with_hooks(
    q_list[a_cl * 100 + b],
    fwd_hooks=[
        (
            f"blocks.{lyr}.hook_resid_post",
            partial(replace_with_clean, clean_acts=get_clean_acts(gpt, (a_cl, b), lyr)),	
        )
    ],
    return_type="logits",
)[:, -1, :]



# %%
arith_acc = arithm_data[1]
print(arith_acc.q_list[43 * 100 + 3])



