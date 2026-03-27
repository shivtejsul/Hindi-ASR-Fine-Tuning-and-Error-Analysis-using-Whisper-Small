"""
Master script to run the entire Hindi ASR assignment pipeline
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_question_1():
    """Run Question 1: Whisper Fine-tuning & Error Analysis"""
    logger.info("\n" + "="*60)
    logger.info("QUESTION 1: Whisper Fine-tuning & Error Analysis")
    logger.info("="*60)
    
    from question_1_whisper_finetuning.q1_solution import main as q1_main
    # q1_main()


def run_question_2():
    """Run Question 2: ASR Cleanup Pipeline"""
    logger.info("\n" + "="*60)
    logger.info("QUESTION 2: ASR Output Cleanup Pipeline")
    logger.info("="*60)
    
    from question_2_cleanup_pipeline.q2_solution import main as q2_main
    # q2_main()


def run_question_3():
    """Run Question 3: Spelling Error Classification"""
    logger.info("\n" + "="*60)
    logger.info("QUESTION 3: Spelling Error Classification")
    logger.info("="*60)
    
    from question_3_spelling_classification.q3_solution import main as q3_main
    # q3_main()


def run_question_4():
    """Run Question 4: Lattice-based WER Evaluation"""
    logger.info("\n" + "="*60)
    logger.info("QUESTION 4: Lattice-based WER Evaluation")
    logger.info("="*60)
    
    from question_4_lattice_wer.q4_solution import main as q4_main
    # q4_main()


def main():
    """Main entry point"""
    logger.info("Starting Hindi ASR Assignment Pipeline")
    logger.info(f"Working directory: {Path.cwd()}")
    
    # Run all questions
    # Uncomment to run each question
    
    # run_question_1()
    # run_question_2()
    # run_question_3()
    # run_question_4()
    
    logger.info("\n" + "="*60)
    logger.info("Assignment pipeline ready!")
    logger.info("="*60)
    logger.info("""
    To run individual questions, uncomment the function calls in main().
    
    Or import and run directly:
    
    from question_1_whisper_finetuning.q1_solution import main as q1_main
    q1_main()
    """)


if __name__ == "__main__":
    main()
