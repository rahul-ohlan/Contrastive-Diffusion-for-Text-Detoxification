"""
Generate detoxified text samples from a trained model and evaluate them.
"""
import os, json
from typing import List, Dict
import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed, AutoTokenizer
import wandb

from src.utils import dist_util, logger
from src.utils.args_utils import create_argparser, args_to_dict, model_and_diffusion_defaults
from src.train_infer.factory_methods import create_model_and_diffusion
from src.evaluation.detox_metrics import DetoxificationEvaluator

def main():
    args = create_argparser().parse_args()

    set_seed(args.seed)
    dist_util.setup_dist()
    logger.configure()

    # Load configurations
    args.checkpoint_path = os.path.split(args.model_name_or_path)[0]
    config_path = os.path.join(args.checkpoint_path, "training_args.json")
    training_args = read_training_args(config_path)
    training_args["batch_size"] = args.batch_size
    training_args["diffusion_steps"] = args.diffusion_steps
    training_args['model_name_or_path'] = args.model_name_or_path
    training_args["clamp"] = args.clamp
    training_args['out_dir'] = args.out_dir
    training_args['num_samples'] = args.num_samples
    args.__dict__.update(training_args)
    args.sigma_small = True

    logger.info(f"Init pretrained = {args.init_pretrained}")
    logger.info(f"Freeze embeddings = {args.freeze_embeddings}")
    logger.info(f"Use pretrained embeddings = {args.use_pretrained_embeddings}")
    
    # Create model and diffusion
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_name_or_path, map_location="cpu"))
    model.eval()

    # Initialize BERT tokenizer for toxic text
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"the parameter count is {pytorch_total_params}")

    diffusion.rescale_timesteps = True

    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    logger.log(f"Clamping is set to {args.clamp}")
    
    # Read toxic sentences from input file
    with open(args.input_file, 'r') as f:
        toxic_sentences = [line.strip() for line in f]
    
    all_samples = []
    all_toxic = []
    
    for i in range(0, len(toxic_sentences), args.batch_size):
        batch_toxic = toxic_sentences[i:i + args.batch_size]
        
        # Tokenize toxic text
        toxic_tokens = tokenizer(
            batch_toxic,
            max_length=args.sequence_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare conditioning
        model_kwargs = {
            "toxic_ids": toxic_tokens['input_ids'].to(dist_util.dev()),
            "toxic_mask": toxic_tokens['attention_mask'].to(dist_util.dev())
        }
        
        # Sample
        sample_shape = (len(batch_toxic), args.sequence_len, model.word_embedding.weight.shape[1])
        sample = diffusion.p_sample_loop(
            model,
            sample_shape,
            clip_denoised=args.clip_denoised,
            denoised_fn=None,
            model_kwargs=model_kwargs,
            top_p=args.top_p,
            progress=True,
            tokenizer=tokenizer
        )

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])
        all_toxic.extend(batch_toxic)

        logger.log(f"created {len(all_samples) * args.batch_size} samples")

    arr = np.concatenate(all_samples, axis=0)
    arr = arr[: args.num_samples]

    x_t = th.tensor(arr).cuda()
    logits = model.get_logits(x_t)
    cands = th.topk(logits, k=1, dim=-1)

    decoded_sentences = []
    for seq in cands.indices:
        decoded_sentence = tokenizer.decode(seq.squeeze(1).tolist(), skip_special_tokens=True)
        decoded_sentences.append(decoded_sentence)

    dist.barrier()
    logger.log("sampling complete")

    # Write outputs
    output_files = write_outputs(args=args, toxic_sentences=all_toxic[:args.num_samples], clean_sentences=decoded_sentences)
    
    # Evaluate outputs
    logger.log("evaluating detoxification...")
    evaluator = DetoxificationEvaluator(device=dist_util.dev())
    metrics = evaluator.evaluate(
        original_texts=all_toxic[:args.num_samples],
        generated_texts=decoded_sentences,
        batch_size=args.batch_size
    )
    
    # Log metrics
    logger.log("=== Detoxification Metrics ===")
    logger.log(f"Style Transfer Accuracy: {metrics['style_transfer']:.4f}")
    logger.log(f"Content Preservation: {metrics['content_preservation']:.4f}")
    logger.log(f"Fluency: {metrics['fluency']:.4f}")
    
    # Save detailed metrics
    metrics_file = output_files["base_path"] + ".metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.log(f"Saved detailed metrics to {metrics_file}")
    
    # Log to wandb if enabled
    if args.use_wandb:
        wandb.init(project="text-detoxification", name=f"eval_{os.path.basename(args.model_name_or_path)}")
        wandb.log({
            "style_transfer": metrics["style_transfer"],
            "content_preservation": metrics["content_preservation"],
            "fluency": metrics["fluency"],
            "samples": wandb.Table(
                columns=["toxic", "clean", "toxicity", "similarity", "fluency"],
                data=[
                    [t, c, ts, ss, fs] for t, c, ts, ss, fs in zip(
                        all_toxic[:args.num_samples],
                        decoded_sentences,
                        metrics["toxicity_scores"],
                        metrics["similarity_scores"],
                        metrics["fluency_scores"]
                    )
                ]
            )
        })
        wandb.finish()

def read_training_args(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def write_outputs(args: dict, toxic_sentences: List[str], clean_sentences: List[str]) -> Dict[str, str]:
    """Write both toxic and detoxified sentences to output files."""
    model_dir = os.path.split(args.model_name_or_path)[0]
    model_base_name = os.path.split(args.model_name_or_path)[1]
    
    base_path = os.path.join(
        model_dir,
        f"{model_base_name}.samples_{len(clean_sentences)}.steps-{args.diffusion_steps}.clamp-{args.clamp}"
    )
    
    # Write paired output
    paired_file = base_path + ".paired.txt"
    with open(paired_file, "w") as f:
        for toxic, clean in zip(toxic_sentences, clean_sentences):
            f.write(f"Toxic: {toxic}\nClean: {clean}\n\n")
    
    # Write clean output only
    clean_file = base_path + ".clean.txt"
    with open(clean_file, "w") as f:
        for sentence in clean_sentences:
            f.write(sentence + "\n")

    print(f"Written the outputs to {paired_file} and {clean_file}")
    
    return {
        "base_path": base_path,
        "paired_file": paired_file,
        "clean_file": clean_file
    }

if __name__ == "__main__":
    main() 