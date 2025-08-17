#!/usr/bin/env python3
import json
import re
import time
import argparse
from typing import Dict, List, Optional, Tuple, Protocol
from abc import ABC, abstractmethod
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModel
from PIL import Image
import numpy as np
from io import BytesIO
import cairosvg
import clip
import os
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class EvalConfig:
    model_type: str  # "diffusion" or "qwen"
    model_path: str
    dataset_path: str
    output_dir: str
    batch_size: int = 8
    max_samples: Optional[int] = None

@dataclass
class GenerationConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.95
    do_sample: bool = True

@dataclass
class DiffusionConfig(GenerationConfig):
    steps: int = 256
    alg: str = "entropy"
    alg_temp: float = 0.0
    output_history: bool = True
    return_dict_in_generate: bool = True

class SVGModel(ABC):
    """Abstract base class for SVG generation models."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    @abstractmethod
    def _load_model(self):
        """Load the model and tokenizer."""
        pass
    
    @abstractmethod
    def generate_svg(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Generate SVG from prompt."""
        pass
    
    @abstractmethod
    def generate_svg_batch(self, prompts: List[str], config: Optional[GenerationConfig] = None) -> List[str]:
        """Generate SVGs from a batch of prompts."""
        pass
    
    @abstractmethod
    def format_prompt(self, prompt: str) -> str:
        """Format the prompt for the specific model."""
        pass

class DiffusionSVGModel(SVGModel):
    """Discrete diffusion model for SVG generation."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__(model_path, device)
    
    def _load_model(self):
        """Load diffusion model and tokenizer."""
        self.model = AutoModel.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.to(self.device)


    def format_prompt(self, prompt: str) -> str:
        """Format prompt for diffusion model."""
        return f"""<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
{prompt.strip()}<|im_end|>
<|im_start|>assistant
"""
    
    def generate_svg(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Generate SVG using discrete diffusion model."""
        if config is None:
            config = DiffusionConfig()
        
        # Convert to DiffusionConfig if needed
        if not isinstance(config, DiffusionConfig):
            diffusion_config = DiffusionConfig(
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample
            )
        else:
            diffusion_config = config
        
        with torch.no_grad():
            full_prompt = self.format_prompt(prompt)
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)

            output = self.model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=diffusion_config.max_new_tokens,
                output_history=diffusion_config.output_history,
                return_dict_in_generate=diffusion_config.return_dict_in_generate,
                steps=diffusion_config.steps,
                temperature=diffusion_config.temperature,
                top_p=diffusion_config.top_p,
                alg=diffusion_config.alg,
                alg_temp=diffusion_config.alg_temp,
            )
            
            generation = self.tokenizer.decode(output.sequences[0][len(input_ids[0]):].tolist())
        return generation.split('<|dlm_pad|>')[0]
    
    def generate_svg_batch(self, prompts: List[str], config: Optional[GenerationConfig] = None) -> List[str]:
        """Generate SVGs using discrete diffusion model with batching."""
        if config is None:
            config = DiffusionConfig()
        
        # Convert to DiffusionConfig if needed
        if not isinstance(config, DiffusionConfig):
            diffusion_config = DiffusionConfig(
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample
            )
        else:
            diffusion_config = config
        
        results = []
        with torch.no_grad():
            # Format all prompts
            full_prompts = [self.format_prompt(prompt) for prompt in prompts]
            
            # Tokenize batch
            inputs = self.tokenizer(full_prompts, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            
            # Generate for the batch
            output = self.model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=diffusion_config.max_new_tokens,
                output_history=diffusion_config.output_history,
                return_dict_in_generate=diffusion_config.return_dict_in_generate,
                steps=diffusion_config.steps,
                temperature=diffusion_config.temperature,
                top_p=diffusion_config.top_p,
                alg=diffusion_config.alg,
                alg_temp=diffusion_config.alg_temp,
            )
            
            # Decode each sequence
            for i, sequence in enumerate(output.sequences):
                original_length = len(input_ids[i])
                generated_tokens = sequence[original_length:].tolist()
                generation = self.tokenizer.decode(generated_tokens)
                results.append(generation.split('<|dlm_pad|>')[0])
        
        return results

class QwenSVGModel(SVGModel):
    """Qwen model for SVG generation."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__(model_path, device)
    
    def _load_model(self):
        """Load Qwen model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path).to(self.device)
        
    
    def format_prompt(self, prompt: str) -> str:
        """Format prompt for Qwen model."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates SVG code."},
            {"role": "user", "content": prompt}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    def generate_svg(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Generate SVG using Qwen model."""
        if config is None:
            config = GenerationConfig()
        
        text = self.format_prompt(prompt)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample
            )
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    def generate_svg_batch(self, prompts: List[str], config: Optional[GenerationConfig] = None) -> List[str]:
        """Generate SVGs using Qwen model with batching."""
        if config is None:
            config = GenerationConfig()
        
        # Format all prompts
        texts = [self.format_prompt(prompt) for prompt in prompts]
        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract only the newly generated tokens for each sequence
        results = []
        for i, (input_ids, output_ids) in enumerate(zip(model_inputs.input_ids, generated_ids)):
            generated_tokens = output_ids[len(input_ids):]
            decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            results.append(decoded)
        
        return results

class SVGProcessor:
    """Handles SVG processing operations."""
    
    @staticmethod
    def extract_svg_from_response(response_text: str) -> Optional[str]:
        """Extract SVG using SVGenius's exact method."""
        # First try to find "Answer: <svg>..." pattern
        answer_pattern = r'Answer:\s*(<svg[\s\S]*?<\/svg>)'
        answer_matches = re.findall(answer_pattern, response_text)
        
        if answer_matches:
            return answer_matches[-1]  
        
        # Fallback to any SVG pattern
        svg_pattern = r'(<svg[\s\S]*?<\/svg>)'
        svg_matches = re.findall(svg_pattern, response_text)
        
        if svg_matches:
            return svg_matches[-1]  
        
        return None

    @staticmethod
    def rasterize_svg(svg_content: str, size: Tuple[int, int] = (224, 224)) -> Optional[Image.Image]:
        """Convert SVG to raster image. Returns None if invalid SVG."""
        try:
            png_data = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), 
                                      output_width=size[0], output_height=size[1])
            return Image.open(BytesIO(png_data)).convert('RGB')
        except Exception:
            return None

class CLIPScorer:
    """Handles CLIP-based scoring operations."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
    
    def calculate_rclip_score(self, image: Image.Image, text: str) -> float:
        """Calculate rCLIP score between image and text."""
        try:
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            text_input = clip.tokenize([text]).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_input)
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarity = (image_features * text_features).sum().item()
                return max(0, similarity)
        except Exception:
            return 0.0

class ModelFactory:
    """Factory class for creating SVG models."""
    
    @staticmethod
    def create_model(model_type: str, model_path: str, device: str = "cuda") -> SVGModel:
        """Create an SVG model based on type."""
        if model_type.lower() == "diffusion":
            return DiffusionSVGModel(model_path, device)
        elif model_type.lower() == "qwen":
            return QwenSVGModel(model_path, device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

class SVGEvaluator:
    """Main evaluation class that orchestrates the evaluation process."""
    
    def __init__(self, config: EvalConfig, generation_config: Optional[GenerationConfig] = None):
        self.config = config
        self.generation_config = generation_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize components
        self.svg_model = ModelFactory.create_model(config.model_type, config.model_path, self.device)
        self.svg_processor = SVGProcessor()
        self.clip_scorer = CLIPScorer(self.device)
        self.results_manager = ResultsManager(config.output_dir, config.model_type)
        
        os.makedirs(config.output_dir, exist_ok=True)

    def evaluate_sample(self, sample: Dict) -> Dict:
        """Evaluate a single sample using SVGenius methodology."""
        instruction = sample["instruction"]
        input_text = sample["input"]
        ground_truth_svg = sample["output"]
        
        prompt = f"{instruction}: {input_text}"
        
        # Generate SVG
        start_time = time.time()
        response = self.svg_model.generate_svg(prompt, self.generation_config)
        generation_time = time.time() - start_time
        
        # Extract SVG using SVGenius method
        generated_svg = self.svg_processor.extract_svg_from_response(response)
        
        result = {
            "prompt": prompt,
            "response": response,
            "generated_svg": generated_svg,
            "ground_truth_svg": ground_truth_svg,
            "generation_time": generation_time,
            "valid_svg": False,
            "clip_score": 0.0,
            "gt_clip_score": 0.0,
            "rclip_score": 0.0
        }
        
        # Test validity by attempting to rasterize (SVGenius approach)
        if generated_svg:
            image = self.svg_processor.rasterize_svg(generated_svg)
            if image:
                result["valid_svg"] = True
                result["clip_score"] = self.clip_scorer.calculate_rclip_score(image, input_text)
        
        # Calculate ground truth rCLIP score
        if ground_truth_svg:
            gt_image = self.svg_processor.rasterize_svg(ground_truth_svg)
            if gt_image:
                result["gt_clip_score"] = self.clip_scorer.calculate_rclip_score(gt_image, input_text)
        
        if result["gt_clip_score"] > 0 and result["valid_svg"]:
            result["rclip_score"] = 1 - max(0, (result["gt_clip_score"] - result["clip_score"]) / result["gt_clip_score"])
        
        return result
    
    def evaluate_batch(self, samples: List[Dict]) -> List[Dict]:
        """Evaluate a batch of samples using SVGenius methodology with batching."""
        # Prepare prompts for batch generation
        prompts = []
        for sample in samples:
            instruction = sample["instruction"]
            input_text = sample["input"]
            prompt = f"{instruction}: {input_text}"
            prompts.append(prompt)
        
        # Generate SVGs in batch
        start_time = time.time()
        responses = self.svg_model.generate_svg_batch(prompts, self.generation_config)
        total_generation_time = time.time() - start_time
        
        # Process each sample
        results = []
        for i, (sample, prompt, response) in enumerate(zip(samples, prompts, responses)):
            input_text = sample["input"]
            ground_truth_svg = sample["output"]
            
            # Estimate individual generation time (rough approximation)
            generation_time = total_generation_time / len(samples)
            
            # Extract SVG using SVGenius method
            generated_svg = self.svg_processor.extract_svg_from_response(response)
            
            result = {
                "prompt": prompt,
                "response": response,
                "generated_svg": generated_svg,
                "ground_truth_svg": ground_truth_svg,
                "generation_time": generation_time,
                "valid_svg": False,
                "clip_score": 0.0,
                "gt_clip_score": 0.0,
                "rclip_score": 0.0
            }
            
            # Test validity by attempting to rasterize (SVGenius approach)
            if generated_svg:
                image = self.svg_processor.rasterize_svg(generated_svg)
                if image:
                    result["valid_svg"] = True
                    result["clip_score"] = self.clip_scorer.calculate_rclip_score(image, input_text)
            
            # Calculate ground truth rCLIP score
            if ground_truth_svg:
                gt_image = self.svg_processor.rasterize_svg(ground_truth_svg)
                if gt_image:
                    result["gt_clip_score"] = self.clip_scorer.calculate_rclip_score(gt_image, input_text)
            
            if result["gt_clip_score"] > 0 and result["valid_svg"]:
                result["rclip_score"] = 1 - max(0, (result["gt_clip_score"] - result["clip_score"]) / result["gt_clip_score"])
            
            results.append(result)
        
        return results

    def run_evaluation(self) -> Dict:
        """Run full evaluation on dataset with batching support."""
        print(f"Loading dataset from {self.config.dataset_path}...")
        with open(self.config.dataset_path, 'r') as f:
            dataset = json.load(f)
        if self.config.max_samples:
            dataset = dataset[:self.config.max_samples]
        
        results = []
        total_samples = len(dataset)
        batch_size = self.config.batch_size
        total_batches = (total_samples + batch_size - 1) // batch_size
        
        print(f"Evaluating {total_samples} samples with {self.config.model_type} model using batch size {batch_size}...")
        print(f"Total batches: {total_batches}")
        
        # Create progress bar
        pbar = tqdm(total=total_batches, desc="Processing batches", unit="batch")
        
        # Process in batches
        for batch_idx in range(0, total_samples, batch_size):
            batch = dataset[batch_idx:batch_idx + batch_size]
            batch_end = min(batch_idx + batch_size, total_samples)
            current_batch_num = batch_idx // batch_size + 1
            
            try:
                batch_results = self.evaluate_batch(batch)
                results.extend(batch_results)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'samples': f"{len(results)}/{total_samples}",
                    'valid_rate': f"{sum(1 for r in results if r['valid_svg'])/len(results):.2%}" if results else "0%"
                })
                
                # Log statistics every 20 batches
                if current_batch_num % 20 == 0 or batch_end == total_samples:
                    self._log_interim_statistics(results, current_batch_num, total_batches)
                
                # Save intermediate results
                if len(results) % (10 * batch_size) == 0 or batch_end == total_samples:
                    self.results_manager.save_results(results, f"intermediate_{len(results)}")
                    
            except Exception as e:
                print(f"\nError processing batch {current_batch_num}: {e}")
                # Fallback to individual processing for this batch
                print("Falling back to individual sample processing for this batch...")
                for j, sample in enumerate(batch):
                    try:
                        result = self.evaluate_sample(sample)
                        results.append(result)
                    except Exception as sample_e:
                        print(f"Error processing sample {batch_idx+j+1}: {sample_e}")
                        continue
                
                # Still update progress bar even on error
                pbar.update(1)
                pbar.set_postfix({
                    'samples': f"{len(results)}/{total_samples}",
                    'valid_rate': f"{sum(1 for r in results if r['valid_svg'])/len(results):.2%}" if results else "0%"
                })
        
        pbar.close()
        
        stats = StatisticsCalculator.calculate_statistics(results)
        
        final_output = {
            "config": self.config.__dict__,
            "results": results,
            "statistics": stats
        }
        
        self.results_manager.save_results(final_output, "final")
        return final_output

    def _log_interim_statistics(self, results: List[Dict], current_batch: int, total_batches: int):
        """Log interim statistics every 20 batches."""
        if not results:
            return
            
        valid_svgs = sum(1 for r in results if r["valid_svg"])
        total = len(results)
        valid_rate = valid_svgs / total if total > 0 else 0
        
        rclip_scores = [r["rclip_score"] for r in results if r["valid_svg"]]
        mean_rclip = np.mean(rclip_scores) if rclip_scores else 0
        
        generation_times = [r["generation_time"] for r in results]
        mean_time = np.mean(generation_times)
        
        print(f"\n{'='*60}")
        print(f"INTERIM STATISTICS - Batch {current_batch}/{total_batches}")
        print(f"{'='*60}")
        print(f"Samples processed: {total}")
        print(f"Valid SVGs: {valid_svgs} ({valid_rate:.2%})")
        print(f"Mean rCLIP score: {mean_rclip:.3f}")
        print(f"Mean generation time: {mean_time:.2f}s")
        print(f"{'='*60}\n")

class StatisticsCalculator:
    """Handles calculation and display of evaluation statistics."""
    
    @staticmethod
    def calculate_statistics(results: List[Dict]) -> Dict:
        """Calculate evaluation statistics."""
        total = len(results)
        valid_svgs = sum(1 for r in results if r["valid_svg"])
        
        rclip_scores = [r["rclip_score"] for r in results if r["valid_svg"]]
        gt_rclip_scores = [r["gt_clip_score"] for r in results if r["gt_clip_score"] > 0]
        generation_times = [r["generation_time"] for r in results]
        
        stats = {
            "total_samples": total,
            "valid_svg_count": valid_svgs,
            "valid_svg_rate": valid_svgs / total if total > 0 else 0,
            "mean_rclip_score": np.mean(rclip_scores) if rclip_scores else 0,
            "std_rclip_score": np.std(rclip_scores) if rclip_scores else 0,
            "mean_gt_rclip_score": np.mean(gt_rclip_scores) if gt_rclip_scores else 0,
            "mean_generation_time": np.mean(generation_times),
            "std_generation_time": np.std(generation_times)
        }
        
        StatisticsCalculator.print_statistics(stats)
        return stats
    
    @staticmethod
    def print_statistics(stats: Dict):
        """Print formatted statistics."""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Total samples: {stats['total_samples']}")
        print(f"Valid SVGs: {stats['valid_svg_count']} ({stats['valid_svg_rate']:.2%})")
        print(f"Mean rCLIP score: {stats['mean_rclip_score']:.3f} ± {stats['std_rclip_score']:.3f}")
        print(f"Mean GT rCLIP score: {stats['mean_gt_rclip_score']:.3f}")
        print(f"Mean generation time: {stats['mean_generation_time']:.2f}s ± {stats['std_generation_time']:.2f}s")
        print("="*50)

class ResultsManager:
    """Handles saving and loading of evaluation results."""
    
    def __init__(self, output_dir: str, model_type: str):
        self.output_dir = output_dir
        self.model_type = model_type
    
    def save_results(self, data, suffix=""):
        """Save results to JSON file."""
        filename = f"evaluation_results_{self.model_type}_{suffix}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {filepath}")
    
    def load_results(self, suffix=""):
        """Load results from JSON file."""
        filename = f"evaluation_results_{self.model_type}_{suffix}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Results file not found: {filepath}")
            return None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SVG Generation Model Evaluation")
    
    # Model configuration
    parser.add_argument("--model_type", type=str, default="diffusion", 
                       choices=["diffusion", "qwen"],
                       help="Type of model to evaluate (default: diffusion)")
    parser.add_argument("--model_path", type=str, 
                       default="saves/diffucoder-svg-lora-3-8-25/checkpoint-1319",
                       help="Path to the model (default: saves/diffucoder-svg-lora-3-8-25/checkpoint-1319)")
     
    # Dataset configuration
    parser.add_argument("--dataset_path", type=str, 
                       default="my_data/omnisvg_only_less_than_1024_tokens_test.json",
                       help="Path to the evaluation dataset (default: my_data/omnisvg_only_less_than_1024_tokens_test.json)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (default: None)")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                       help="Directory to save evaluation results (default: ./eval_results)")
    
    # Processing configuration
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for processing (default: 8)")
    
    # Generation configuration
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                       help="Maximum number of new tokens to generate (default: 1024)")
    parser.add_argument("--temperature", type=float, default=0.3,
                       help="Temperature for generation (default: 0.3)")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Top-p value for generation (default: 0.95)")
    parser.add_argument("--do_sample", action="store_true", default=True,
                       help="Whether to use sampling for generation (default: True)")
    
    # Diffusion-specific configuration
    parser.add_argument("--steps", type=int, default=1024,
                       help="Number of diffusion steps (default: 1024)")
    parser.add_argument("--alg", type=str, default="entropy",
                       help="Diffusion algorithm (default: entropy)")
    parser.add_argument("--alg_temp", type=float, default=0.0,
                       help="Algorithm temperature (default: 0.0)")
    parser.add_argument("--output_history", action="store_true", default=True,
                       help="Whether to output history (default: True)")
    parser.add_argument("--return_dict_in_generate", action="store_true", default=True,
                       help="Whether to return dict in generate (default: True)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create configuration based on parsed arguments
    config = EvalConfig(
        model_type=args.model_type,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    # Adjust output directory based on model type if using default
    if args.output_dir == "./eval_results" and args.model_type == "diffusion":
        config.output_dir = "./eval_results_diffusion"
    # add to output directory the time
    config.output_dir = os.path.join(config.output_dir, time.strftime("%Y%m%d_%H%M%S"))
    print(f"Running evaluation with the following configuration:")
    print(f"Model type: {config.model_type}")
    print(f"Model path: {config.model_path}")
    print(f"Dataset path: {config.dataset_path}")
    print(f"Output directory: {config.output_dir}")
    print(f"Batch size: {config.batch_size}")
    print(f"Max samples: {config.max_samples}")
    print(f"Generation parameters:")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Do sample: {args.do_sample}")
    
    if args.model_type == "diffusion":
        print(f"Diffusion parameters:")
        print(f"  Steps: {args.steps}")
        print(f"  Algorithm: {args.alg}")
        print(f"  Algorithm temperature: {args.alg_temp}")
        print(f"  Output history: {args.output_history}")
        print(f"  Return dict in generate: {args.return_dict_in_generate}")

    # Create generation config based on model type
    if args.model_type == "diffusion":
        generation_config = DiffusionConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            steps=args.steps,
            alg=args.alg,
            alg_temp=args.alg_temp,
            output_history=args.output_history,
            return_dict_in_generate=args.return_dict_in_generate
        )
    else:
        generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample
        )

    evaluator = SVGEvaluator(config, generation_config)
    
    results = evaluator.run_evaluation()