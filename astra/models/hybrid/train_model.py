import argparse

from astra.utils import logger
from astra.evaluation.hybrid_model import run_eval

from astra.data.dataloader import prepare_data_and_dls
from astra.models.hybrid.training import run_pretrain, run_finetune


def parse_args():
    parser = argparse.ArgumentParser(description='MLM Pretraining + Fine-tuning Pipeline')
    
    parser.add_argument('--model-name', '-m', default='10122025', help='Model name')
    parser.add_argument('--lr', type=float, default=4.7863e-4, help='Learning rate')
    parser.add_argument('--finetune-epochs', type=int, default=22, help='Fine-tune epochs')
    
    # Stage flags
    parser.add_argument('--pretrain', action='store_true', default=True)
    parser.add_argument('--no-pretrain', dest='pretrain', action='store_false')
    parser.add_argument('--finetune', action='store_true', default=True)
    parser.add_argument('--no-finetune', dest='finetune', action='store_false')
    parser.add_argument('--eval', action='store_true', default=True)
    parser.add_argument('--no-eval', dest='eval', action='store_false')
    
    # New eval flag
    parser.add_argument('--comprehensive-eval', action='store_true', default=True,
                       help='Run full time-dependent evaluation (default)')
    parser.add_argument('--simple-eval', dest='comprehensive_eval', action='store_false',
                       help='Run simple single-point evaluation')
    
    parser.add_argument('--use-pretrained', action='store_true', default=True)
    parser.add_argument('--skip-valid', action='store_true', default=True)
    parser.add_argument('--device', default='cuda')
    
    return parser.parse_args()

def main():
    args = parse_args()
    data = prepare_data_and_dls()
    pretrain_cfg = None
    
    if args.pretrain:
        logger.info("=== Running Pretraining ===")
        pretrain_cfg, _, _ = run_pretrain(data, device=args.device)
    
    if args.finetune:
        logger.info("=== Running Fine-tuning ===")
        run_finetune(data, args.model_name, args.use_pretrained, pretrain_cfg,
                    args.skip_valid, args.lr, args.finetune_epochs)
    
    if args.eval:
        logger.info("=== Running Evaluation ===")
        results = run_eval(data, args.model_name, args.comprehensive_eval)
        if args.comprehensive_eval:
            logger.info(f"Generated {len(results[0])} time points with CIs")

if __name__ == "__main__":
    main()