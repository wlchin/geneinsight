#!/usr/bin/env python3
"""
Main pipeline module for the TopicGenes package.
"""

import os
import time
import logging
import tempfile
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class Pipeline:
    """Main pipeline for processing gene sets through topic modeling and enrichment."""
    
    def __init__(
        self,
        output_dir: str,
        temp_dir: Optional[str] = "temp",  # Changed default to "temp"
        n_samples: int = 5,
        num_topics: Optional[int] = 10,
        pvalue_threshold: float = 0.01,
        api_service: str = "openai",
        api_model: str = "gpt-4o-mini",
        api_parallel_jobs: int = 4,
        api_base_url: Optional[str] = None,
        target_filtered_topics: int = 25,
    ):
        """
        Initialize the pipeline.
        
        Args:
            output_dir: Directory to store final outputs
            temp_dir: Directory for temporary files (if None, uses system temp dir)
            n_samples: Number of topic models to run with different seeds
            num_topics: Number of topics to extract in topic modeling
            pvalue_threshold: Adjusted P-value threshold for filtering results
            api_service: API service for topic refinement ("openai", "together", etc.)
            api_model: Model name for the API service
            api_parallel_jobs: Number of parallel API jobs
            api_base_url: Base URL for the API service (if needed)
            target_filtered_topics: Target number of topics after filtering
        """
        self.output_dir = os.path.abspath(output_dir)
        self.temp_dir = os.path.abspath(temp_dir if temp_dir is not None else "temp")  # Fallback to "temp" if None
        self.n_samples = n_samples
        self.num_topics = num_topics
        self.pvalue_threshold = pvalue_threshold
        self.api_service = api_service
        self.api_model = api_model
        self.api_parallel_jobs = api_parallel_jobs
        self.api_base_url = api_base_url
        self.target_filtered_topics = target_filtered_topics
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Setup intermediate output directories
        self.dirs = {
            "enrichment": os.path.join(self.temp_dir, "enrichment"),
            "topics": os.path.join(self.temp_dir, "topics"),
            "prompts": os.path.join(self.temp_dir, "prompts"),
            "minor_topics": os.path.join(self.temp_dir, "minor_topics"),
            "summary": os.path.join(self.temp_dir, "summary"),
            "filtered_sets": os.path.join(self.temp_dir, "filtered_sets"),
            "resampled_topics": os.path.join(self.temp_dir, "resampled_topics"),
            "key_topics": os.path.join(self.temp_dir, "key_topics"),
            "final": os.path.join(self.output_dir, "results"),
        }
        
        # Create all subdirectories
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            
        # Set timestamp for run
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
            
        logger.info(f"Pipeline initialized with output directory: {self.output_dir}")
        logger.info(f"Using temporary directory: {self.temp_dir}")
    
    def run(
        self,
        query_gene_set: str,
        background_gene_list: str,
        zip_output: bool = True,
    ) -> str:
        """
        Run the full pipeline.
        
        Args:
            query_gene_set: Path to file containing query gene set
            background_gene_list: Path to file containing background gene list
            zip_output: Whether to zip the output directory
            
        Returns:
            Path to the output directory or zip file
        """
        logger.info(f"Starting pipeline run with query gene set: {query_gene_set}")
        
        # Extract gene set name from file path
        gene_set_name = os.path.splitext(os.path.basename(query_gene_set))[0]
        run_id = f"{gene_set_name}_{self.timestamp}"
        logger.info(f"Run ID: {run_id}")
        
        # 1. Get gene enrichment data from StringDB
        logger.info("Step 1: Retrieving gene enrichment data from StringDB")
        enrichment_df, documents_df = self._get_stringdb_enrichment(query_gene_set)
        
        # 2. Run topic modeling
        logger.info("Step 2: Running topic modeling")
        topics_df = self._run_topic_modeling(documents_df)
        
        # 3. Generate prompts
        logger.info("Step 3: Generating prompts for topic refinement")
        prompts_df = self._generate_prompts(topics_df)
        
        # 4. Process through API (without caching)
        logger.info("Step 4: Processing prompts through API without caching")
        api_results_df = self._process_api_calls(prompts_df)
        
        # 5. Create summary
        logger.info("Step 5: Creating summary")
        summary_df = self._create_summary(api_results_df, enrichment_df)
        summary_df.to_csv("testing.csv", index=False)
        
        # 6. Perform hypergeometric enrichment
        logger.info("Step 6: Performing hypergeometric enrichment")
        enriched_df = self._perform_hypergeometric_enrichment(
            summary_df, query_gene_set, background_gene_list
        )
        
        # 7. Run topic modeling on the filtered gene sets (meta-analysis)
        logger.info("Step 7: Running topic modeling on filtered gene sets")
        topics_df = self._run_topic_modeling_on_filtered_sets(enriched_df)
        
        # 8. Extract key topics
        logger.info("Step 8: Extracting key topics")
        key_topics_df = self._get_key_topics(topics_df)
        
        # 9. Filter topics by similarity using key topics
        logger.info("Step 9: Filtering topics by similarity")
        filtered_df = self._filter_topics(key_topics_df)
        
        # If the filtered DataFrame has fewer than the target rows (default: 25), 
        # then use the entire enriched DataFrame as input for filtering and clustering.
        if len(filtered_df) < self.target_filtered_topics:
            logger.info("Filtered topics fewer than target; using entire enriched dataframe for filtering and clustering.")
            filtered_df = self._filter_topics(enriched_df)
        
        # 9b. Run clustering on the filtered topics
        logger.info("Step 9b: Clustering filtered topics")
        clustered_df = self._run_clustering(filtered_df)
        
        # 10. Finalize outputs and cleanup
        logger.info("Step 10: Finalizing outputs")
        output_path = self._finalize_outputs(
            run_id,
            {
                "enrichment": enrichment_df,
                "documents": documents_df,
                "topics": topics_df,
                "prompts": prompts_df,
                "api_results": api_results_df,
                "summary": summary_df,
                "enriched": enriched_df,
                "key_topics": key_topics_df,
                "clustered": clustered_df,
            },
            zip_output
        )
        
        logger.info(f"Pipeline completed successfully. Results available at: {output_path}")
        return output_path
    
    def _get_stringdb_enrichment(self, query_gene_set: str) -> tuple:
        """Get gene enrichment data from StringDB."""
        from .enrichment.stringdb import process_gene_enrichment
        
        enrichment_output = os.path.join(self.dirs["enrichment"], "enrichment.csv")
        documents_output = os.path.join(self.dirs["enrichment"], "documents.csv")
        
        enrichment_df, documents = process_gene_enrichment(
            input_file=query_gene_set,
            output_dir=self.dirs["enrichment"],
            mode="single"
        )
        
        return enrichment_df, pd.DataFrame({"description": documents})
    
    def _run_topic_modeling(self, documents_df: pd.DataFrame) -> pd.DataFrame:
        """Run topic modeling on documents."""
        from .models.bertopic import run_multiple_seed_topic_modeling
        
        documents_path = os.path.join(self.dirs["enrichment"], "documents_for_modeling.csv")
        topics_output = os.path.join(self.dirs["topics"], "topics.csv")
        
        # Save documents to CSV first
        documents_df.to_csv(documents_path, index=False)
        
        topics_df = run_multiple_seed_topic_modeling(
            input_file=documents_path,
            output_file=topics_output,
            method="bertopic",
            num_topics=self.num_topics,
            seed_value=42,
            n_samples=self.n_samples,
            use_sentence_embeddings=True
        )
        
        return topics_df
    
    def _generate_prompts(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        """Generate prompts for topic refinement."""
        from .workflows.prompt_generation import generate_prompts
        
        topics_path = os.path.join(self.dirs["topics"], "topics.csv")
        prompts_output = os.path.join(self.dirs["prompts"], "prompts.csv")
        
        # Save topics to CSV first (if not already saved)
        if not os.path.exists(topics_path):
            topics_df.to_csv(topics_path, index=False)
        
        prompts_df = generate_prompts(
            input_file=topics_path,
            num_subtopics=5,
            max_words=10,
            output_file=prompts_output,
            max_retries=3
        )
        
        return prompts_df
    
    def _process_api_calls(self, prompts_df: pd.DataFrame) -> pd.DataFrame:
        """Process prompts through API without caching."""
        from .api.client import batch_process_api_calls

        prompts_path = os.path.join(self.dirs["prompts"], "prompts.csv")
        api_output = os.path.join(self.dirs["minor_topics"], "api_results.csv")
        
        # Save prompts to CSV if not already saved.
        if not os.path.exists(prompts_path):
            prompts_df.to_csv(prompts_path, index=False)
        
        logger.info("Running API calls without caching.")
        api_results_df = batch_process_api_calls(
            prompts_csv=prompts_path,
            output_api=api_output,
            service=self.api_service,
            model=self.api_model,
            base_url=self.api_base_url,
            n_jobs=self.api_parallel_jobs
        )
        return api_results_df
    
    def _create_summary(self, api_results_df: pd.DataFrame, enrichment_df: pd.DataFrame) -> pd.DataFrame:
        """Create summary by combining API results with enrichment data."""
        from .analysis.summary import create_summary
        
        api_path = os.path.join(self.dirs["minor_topics"], "api_results.csv")
        enrichment_path = os.path.join(self.dirs["enrichment"], "enrichment.csv")
        summary_output = os.path.join(self.dirs["summary"], "summary.csv")
        
        # Save to CSV first (if not already saved)
        if not os.path.exists(api_path):
            api_results_df.to_csv(api_path, index=False)
        if not os.path.exists(enrichment_path):
            enrichment_df.to_csv(enrichment_path, index=False)
        
        summary_df = create_summary(api_results_df, enrichment_df, summary_output)
        return summary_df
    
    def _perform_hypergeometric_enrichment(
        self, 
        summary_df: pd.DataFrame, 
        query_gene_set: str, 
        background_gene_list: str
    ) -> pd.DataFrame:
        """Perform hypergeometric enrichment analysis."""
        from .enrichment.hypergeometric import hypergeometric_enrichment
        
        summary_path = os.path.join(self.dirs["summary"], "summary.csv")
        enriched_output = os.path.join(self.dirs["filtered_sets"], "enriched.csv")
        
        # Save summary to CSV if not already saved
        if not os.path.exists(summary_path):
            summary_df.to_csv(summary_path, index=False)
        
        enriched_df = hypergeometric_enrichment(
            df_path=summary_path,
            gene_origin_path=query_gene_set,
            background_genes_path=background_gene_list,
            output_csv=enriched_output,
            pvalue_threshold=self.pvalue_threshold
        )
        
        return enriched_df
    
    def _run_topic_modeling_on_filtered_sets(self, filtered_df: pd.DataFrame) -> pd.DataFrame:
        """Run topic modeling on the filtered gene sets (meta-analysis)."""
        from .models.meta import run_multiple_seed_topic_modeling
        
        temp_filtered_file = os.path.join(self.dirs["filtered_sets"], "temp_filtered_sets.csv")
        filtered_df.to_csv(temp_filtered_file, index=False)
        
        resampled_output = os.path.join(self.dirs["resampled_topics"], "resampled_topics.csv")
        
        logger.info("Running topic modeling on filtered gene sets")
        topics_df = run_multiple_seed_topic_modeling(
            input_file=temp_filtered_file,
            output_file=resampled_output,
            method="bertopic",
            num_topics=None,  # Auto-determine
            n_samples=10
        )
        
        return topics_df
    
    def _get_key_topics(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        """Count the top terms in the topic modeling results."""
        from .analysis.counter import count_top_terms
        
        temp_topics_file = os.path.join(self.dirs["resampled_topics"], "temp_topics.csv")
        topics_df.to_csv(temp_topics_file, index=False)
        
        key_topics_output = os.path.join(self.dirs["key_topics"], "key_topics.csv")
        
        logger.info("Counting key topics")
        key_topics_df = count_top_terms(
            input_file=temp_topics_file,
            output_file=key_topics_output,
            top_n=None  # Get all topics
        )
        logger.info(f"Found {len(key_topics_df)} key topics")
        
        return key_topics_df
    
    def _filter_topics(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Filter topics by similarity."""
        from .analysis.similarity import filter_terms_by_similarity
        
        # Always write the provided input DataFrame to file
        key_topics_file = os.path.join(self.dirs["key_topics"], "key_topics.csv")
        input_df.to_csv(key_topics_file, index=False)
        
        filtered_output = os.path.join(self.dirs["filtered_sets"], "filtered.csv")
        
        logger.info("Filtering topics by similarity")
        filtered_df = filter_terms_by_similarity(
            input_csv=key_topics_file,
            output_csv=filtered_output,
            target_rows=self.target_filtered_topics
        )
        
        return filtered_df
    
    def _run_clustering(self, filtered_df: pd.DataFrame) -> pd.DataFrame:
        """Run clustering on the filtered topics using the geneinsight.analysis.clustering module."""
        from .analysis.clustering import run_clustering  # Assuming this function is available in the module
        
        # Save filtered topics to a CSV file that will be used as input for clustering
        clustering_input = os.path.join(self.dirs["filtered_sets"], "filtered.csv")
        filtered_df.to_csv(clustering_input, index=False)
        
        # Define the output path for the clustering results
        clustering_output = os.path.join(self.dirs["filtered_sets"], "clustered.csv")
        
        # Run the clustering function with desired parameters
        run_clustering(
            input_csv=clustering_input,
            output_csv=clustering_output,
            min_clusters=5,
            max_clusters=10,
            n_trials=100
        )
        
        # Load the clustered data and return it
        clustered_df = pd.read_csv(clustering_output)
        return clustered_df
    
    def _finalize_outputs(
        self,
        run_id: str,
        dataframes: Dict[str, pd.DataFrame],
        zip_output: bool = True
    ) -> str:
        """
        Finalize outputs by copying to the output directory and optionally zipping.
        
        Args:
            run_id: Unique identifier for this run
            dataframes: Dictionary of dataframes to save
            zip_output: Whether to zip the output directory
            
        Returns:
            Path to the output directory or zip file
        """
        from .utils.zip_helper import zip_directory
        
        run_dir = os.path.join(self.dirs["final"], run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        for name, df in dataframes.items():
            output_path = os.path.join(run_dir, f"{name}.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {name} to {output_path}")
        
        metadata = {
            "run_id": run_id,
            "timestamp": self.timestamp,
            "n_samples": self.n_samples,
            "num_topics": self.num_topics,
            "pvalue_threshold": self.pvalue_threshold,
            "api_service": self.api_service,
            "api_model": self.api_model,
        }
        
        metadata_path = os.path.join(run_dir, "metadata.csv")
        pd.DataFrame([metadata]).to_csv(metadata_path, index=False)
        
        if zip_output:
            zip_path = os.path.join(self.output_dir, f"{run_id}.zip")
            zip_directory(run_dir, zip_path)
            logger.info(f"Zipped output directory to {zip_path}")
            return zip_path
        
        return run_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the TopicGenes pipeline.")
    parser.add_argument(
        "-q", "--query_gene_set", required=True, help="Path to file containing query gene set."
    )
    parser.add_argument(
        "-b", "--background_gene_list", required=True, help="Path to file containing background gene list."
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Directory to store final outputs."
    )
    parser.add_argument(
        "--zip_output", action="store_true", help="Whether to zip the output directory."
    )
    parser.add_argument(
        "--n_samples", type=int, default=5, help="Number of topic models to run with different seeds."
    )
    parser.add_argument(
        "--num_topics", type=int, default=None, help="Number of topics to extract in topic modeling."
    )
    parser.add_argument(
        "--pvalue_threshold", type=float, default=0.01, help="Adjusted P-value threshold for filtering results."
    )
    parser.add_argument(
        "--api_service", type=str, default="openai", help="API service for topic refinement."
    )
    parser.add_argument(
        "--api_model", type=str, default="gpt-4o-mini", help="Model name for the API service."
    )
    parser.add_argument(
        "--api_parallel_jobs", type=int, default=4, help="Number of parallel API jobs."
    )
    parser.add_argument(
        "--api_base_url", type=str, default=None, help="Base URL for the API service, if needed."
    )
    parser.add_argument(
        "--target_filtered_topics", type=int, default=25, help="Target number of topics after filtering."
    )
    parser.add_argument(
        "--temp_dir", type=str, default="temp", help="Temporary directory for intermediate files."
    )
    args = parser.parse_args()
    
    pipeline = Pipeline(
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,
        n_samples=args.n_samples,
        num_topics=args.num_topics,
        pvalue_threshold=args.pvalue_threshold,
        api_service=args.api_service,
        api_model=args.api_model,
        api_parallel_jobs=args.api_parallel_jobs,
        api_base_url=args.api_base_url,
        target_filtered_topics=args.target_filtered_topics,
    )
    pipeline.run(args.query_gene_set, args.background_gene_list, zip_output=args.zip_output)