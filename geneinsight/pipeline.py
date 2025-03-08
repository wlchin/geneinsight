import os
import time
import logging
import tempfile
import hashlib
import json
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class Pipeline:
    """Main pipeline for processing gene sets through topic modeling and enrichment with caching."""
    
    def __init__(
        self,
        output_dir: str,
        temp_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        n_samples: int = 5,
        num_topics: Optional[int] = 10,
        pvalue_threshold: float = 0.01,
        api_service: str = "openai",
        api_model: str = "gpt-4o-mini",
        api_parallel_jobs: int = 4,
        api_base_url: Optional[str] = None,
        target_filtered_topics: int = 25,
        use_cache: bool = True,
    ):
        """
        Initialize the pipeline.
        
        Args:
            output_dir: Directory to store final outputs
            temp_dir: Directory for temporary files (if None, uses system temp dir)
            cache_dir: Directory for cached results (if None, uses 'cache' in output_dir)
            n_samples: Number of topic models to run with different seeds
            num_topics: Number of topics to extract in topic modeling
            pvalue_threshold: Adjusted P-value threshold for filtering results
            api_service: API service for topic refinement ("openai", "together", etc.)
            api_model: Model name for the API service
            api_parallel_jobs: Number of parallel API jobs
            api_base_url: Base URL for the API service (if needed)
            target_filtered_topics: Target number of topics after filtering
            use_cache: Whether to use caching functionality
        """
        self.output_dir = os.path.abspath(output_dir)
        self.temp_dir = os.path.abspath(temp_dir) if temp_dir else tempfile.mkdtemp(prefix="topicgenes_")
        self.cache_dir = os.path.abspath(cache_dir) if cache_dir else os.path.join(self.output_dir, "cache")
        self.n_samples = n_samples
        self.num_topics = num_topics
        self.pvalue_threshold = pvalue_threshold
        self.api_service = api_service
        self.api_model = api_model
        self.api_parallel_jobs = api_parallel_jobs
        self.api_base_url = api_base_url
        self.target_filtered_topics = target_filtered_topics
        self.use_cache = use_cache
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
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
            "cache": self.cache_dir,
        }
        
        # Create all subdirectories
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            
        # Set timestamp for run
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
            
        logger.info(f"Pipeline initialized with output directory: {self.output_dir}")
        logger.info(f"Using temporary directory: {self.temp_dir}")
        logger.info(f"Using cache directory: {self.cache_dir}")
        logger.info(f"Cache enabled: {self.use_cache}")
    
    def _generate_cache_key(self, file_path: str, background_path: Optional[str] = None, 
                           additional_params: Optional[Dict] = None) -> str:
        """
        Generate a cache key based on file content hash and optional parameters.
        
        Args:
            file_path: Path to the input file
            background_path: Optional path to background gene list
            additional_params: Optional dict of additional parameters to include in cache key
            
        Returns:
            A unique hash string to use as cache key
        """
        hasher = hashlib.md5()
        
        # Add file content hash
        with open(file_path, 'rb') as f:
            hasher.update(f.read())
        
        # Add background file content hash if provided
        if background_path:
            with open(background_path, 'rb') as f:
                hasher.update(f.read())
        
        # Add additional parameters if provided
        if additional_params:
            param_str = json.dumps(additional_params, sort_keys=True)
            hasher.update(param_str.encode())
            
        return hasher.hexdigest()
    
    def _check_cache(self, cache_key: str, cache_type: str) -> Optional[Dict]:
        """
        Check if a cached result exists.
        
        Args:
            cache_key: The cache key to check
            cache_type: Type of cache (e.g., 'stringdb', 'api_calls')
            
        Returns:
            Dict with cached file paths if cache exists, None otherwise
        """
        if not self.use_cache:
            return None
            
        cache_folder = os.path.join(self.dirs["cache"], cache_type, cache_key)
        
        # Check if cache folder exists
        if not os.path.isdir(cache_folder):
            return None
            
        # Check for cache metadata
        metadata_path = os.path.join(cache_folder, "metadata.json")
        if not os.path.exists(metadata_path):
            return None
            
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Verify all files exist
        for file_key, file_path in metadata.get("files", {}).items():
            if not os.path.exists(file_path):
                logger.warning(f"Cache incomplete: {file_path} missing")
                return None
                
        logger.info(f"Cache hit for {cache_type} with key {cache_key}")
        return metadata
    
    def _save_to_cache(self, cache_key: str, cache_type: str, 
                     files: Dict[str, str], metadata: Optional[Dict] = None) -> str:
        """
        Save results to cache.
        
        Args:
            cache_key: The cache key to use
            cache_type: Type of cache (e.g., 'stringdb', 'api_calls')
            files: Dict mapping file types to file paths
            metadata: Optional additional metadata to store
            
        Returns:
            Path to the cache directory
        """
        if not self.use_cache:
            return None
            
        cache_folder = os.path.join(self.dirs["cache"], cache_type, cache_key)
        os.makedirs(cache_folder, exist_ok=True)
        
        # Copy files to cache
        cached_files = {}
        for file_key, source_path in files.items():
            if not os.path.exists(source_path):
                logger.warning(f"Cannot cache non-existent file: {source_path}")
                continue
                
            cache_path = os.path.join(cache_folder, f"{file_key}.csv")
            
            # Copy the file
            try:
                pd.read_csv(source_path).to_csv(cache_path, index=False)
                cached_files[file_key] = cache_path
            except Exception as e:
                logger.error(f"Error caching file {source_path}: {str(e)}")
        
        # Save metadata
        cache_metadata = {
            "created": time.time(),
            "files": cached_files,
        }
        
        # Add additional metadata if provided
        if metadata:
            cache_metadata.update(metadata)
            
        with open(os.path.join(cache_folder, "metadata.json"), 'w') as f:
            json.dump(cache_metadata, f, indent=2)
            
        logger.info(f"Saved results to cache: {cache_type} with key {cache_key}")
        return cache_folder
    
    def run(
        self,
        query_gene_set: str,
        background_gene_list: str,
        zip_output: bool = True,
    ) -> str:
        """
        Run the full pipeline with caching support.
        
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
        
        # 1. Get gene enrichment data from StringDB (with caching)
        logger.info("Step 1: Retrieving gene enrichment data from StringDB")
        enrichment_df, documents_df = self._get_stringdb_enrichment(query_gene_set)
        
        # 2. Run topic modeling
        logger.info("Step 2: Running topic modeling")
        topics_df = self._run_topic_modeling(documents_df)
        
        # 3. Generate prompts
        logger.info("Step 3: Generating prompts for topic refinement")
        prompts_df = self._generate_prompts(topics_df)
        
        # 4. Process through API (with caching)
        logger.info("Step 4: Processing prompts through API")
        api_results_df = self._process_api_calls(prompts_df, query_gene_set)
        
        # 5. Create summary
        logger.info("Step 5: Creating summary")
        summary_df = self._create_summary(api_results_df, enrichment_df)
        
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
        
        # 9. Filter topics by similarity
        logger.info("Step 9: Filtering topics by similarity")
        filtered_df = self._filter_topics(key_topics_df)
        
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
                "filtered": filtered_df,
            },
            zip_output
        )
        
        logger.info(f"Pipeline completed successfully. Results available at: {output_path}")
        return output_path
    
    def _get_stringdb_enrichment(self, query_gene_set: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get gene enrichment data from StringDB with caching support."""
        from .enrichment.stringdb import process_gene_enrichment
        
        # Generate cache key
        cache_key = self._generate_cache_key(
            query_gene_set, 
            additional_params={"api_service": self.api_service}
        )
        
        # Check if results are cached
        cache_data = self._check_cache(cache_key, "stringdb")
        if cache_data and self.use_cache:
            try:
                logger.info("Using cached StringDB enrichment results")
                
                # Make sure the required files exist in the cache data
                if "files" not in cache_data or "enrichment" not in cache_data["files"] or "documents" not in cache_data["files"]:
                    logger.warning("Cache data structure is incorrect or incomplete")
                    logger.debug(f"Cache data: {cache_data}")
                    # Fall through to non-cached path
                else:
                    # Try to read the cached files
                    enrichment_file = cache_data["files"]["enrichment"]
                    documents_file = cache_data["files"]["documents"]
                    
                    if os.path.exists(enrichment_file) and os.path.exists(documents_file):
                        enrichment_df = pd.read_csv(enrichment_file)
                        documents_df = pd.read_csv(documents_file)
                        
                        # Copy to current temp directory
                        enrichment_output = os.path.join(self.dirs["enrichment"], "enrichment.csv")
                        documents_output = os.path.join(self.dirs["enrichment"], "documents.csv")
                        enrichment_df.to_csv(enrichment_output, index=False)
                        documents_df.to_csv(documents_output, index=False)
                        
                        return enrichment_df, documents_df
                    else:
                        logger.warning(f"Cached files don't exist: {enrichment_file} or {documents_file}")
            except Exception as e:
                logger.warning(f"Error reading cached data: {str(e)}")
                # Fall through to non-cached path
        
        # If not cached, proceed with API call
        logger.info("StringDB results not in cache, making API call")
        enrichment_output = os.path.join(self.dirs["enrichment"], "enrichment.csv")
        documents_output = os.path.join(self.dirs["enrichment"], "documents.csv")
        
        enrichment_df, documents = process_gene_enrichment(
            input_file=query_gene_set,
            output_dir=self.dirs["enrichment"],
            mode="single"
        )
        
        # Create DataFrame from documents if it's just a list
        if isinstance(documents, list):
            documents_df = pd.DataFrame({"description": documents})
        else:
            documents_df = documents
            
        # Save results
        documents_df.to_csv(documents_output, index=False)
        
        # Cache the results
        self._save_to_cache(
            cache_key, 
            "stringdb",
            {"enrichment": enrichment_output, "documents": documents_output},
            {"query_gene_set": query_gene_set}
        )
        
        return enrichment_df, documents_df
    
    def _process_api_calls(self, prompts_df: pd.DataFrame, query_gene_set: str) -> pd.DataFrame:
        """Process prompts through API with caching support."""
        from .api.client import batch_process_api_calls

        prompts_path = os.path.join(self.dirs["prompts"], "prompts.csv")
        api_output = os.path.join(self.dirs["minor_topics"], "api_results.csv")
        
        # Save prompts to CSV if not already saved.
        if not os.path.exists(prompts_path):
            prompts_df.to_csv(prompts_path, index=False)
        
        # Generate cache key based on prompts content and API parameters
        cache_key = self._generate_cache_key(
            prompts_path, 
            additional_params={
                "api_service": self.api_service,
                "api_model": self.api_model,
                "query_gene_set": os.path.basename(query_gene_set)
            }
        )
        
        # Check if results are cached
        cache_data = self._check_cache(cache_key, "api_calls")
        if cache_data and self.use_cache:
            try:
                logger.info("Using cached API results")
                
                # Make sure the required files exist in the cache data
                if "files" not in cache_data or "api_results" not in cache_data["files"]:
                    logger.warning("API cache data structure is incorrect or incomplete")
                    logger.debug(f"Cache data: {cache_data}")
                    # Fall through to non-cached path
                else:
                    # Try to read the cached files
                    api_results_file = cache_data["files"]["api_results"]
                    
                    if os.path.exists(api_results_file):
                        api_results_df = pd.read_csv(api_results_file)
                        
                        # Copy to current temp directory
                        api_results_df.to_csv(api_output, index=False)
                        
                        return api_results_df
                    else:
                        logger.warning(f"Cached API results file doesn't exist: {api_results_file}")
            except Exception as e:
                logger.warning(f"Error reading cached API data: {str(e)}")
                # Fall through to non-cached path
        
        # If not cached, proceed with API calls
        logger.info("API results not in cache, making API calls")
        api_results_df = batch_process_api_calls(
            prompts_csv=prompts_path,
            output_api=api_output,
            service=self.api_service,
            model=self.api_model,
            base_url=self.api_base_url,
            n_jobs=self.api_parallel_jobs
        )
        
        # Cache the results
        self._save_to_cache(
            cache_key, 
            "api_calls",
            {"api_results": api_output, "prompts": prompts_path},
            {
                "api_service": self.api_service,
                "api_model": self.api_model,
                "query_gene_set": os.path.basename(query_gene_set)
            }
        )
        
        return api_results_df

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
        
        return key_topics_df
    
    def _filter_topics(self, key_topics_df: pd.DataFrame) -> pd.DataFrame:
        """Filter topics by similarity."""
        from .analysis.similarity import filter_terms_by_similarity
        
        key_topics_file = os.path.join(self.dirs["key_topics"], "key_topics.csv")
        if not os.path.exists(key_topics_file):
            key_topics_df.to_csv(key_topics_file, index=False)
        
        filtered_output = os.path.join(self.dirs["filtered_sets"], "filtered.csv")
        
        logger.info("Filtering topics by similarity")
        filtered_df = filter_terms_by_similarity(
            input_csv=key_topics_file,
            output_csv=filtered_output,
            target_rows=self.target_filtered_topics
        )
        
        return filtered_df
    
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
        "--num_topics", type=int, default=10, help="Number of topics to extract in topic modeling."
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
        "--temp_dir", type=str, default=None, help="Temporary directory for intermediate files."
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="Directory for cached results."
    )
    parser.add_argument(
        "--use_cache", action="store_true", default=True, help="Whether to use caching functionality."
    )
    parser.add_argument(
        "--no_cache", action="store_false", dest="use_cache", help="Disable caching functionality."
    )
    args = parser.parse_args()
    
    pipeline = Pipeline(
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,
        cache_dir=args.cache_dir,
        n_samples=args.n_samples,
        num_topics=args.num_topics,
        pvalue_threshold=args.pvalue_threshold,
        api_service=args.api_service,
        api_model=args.api_model,
        api_parallel_jobs=args.api_parallel_jobs,
        api_base_url=args.api_base_url,
        target_filtered_topics=args.target_filtered_topics,
        use_cache=args.use_cache,
    )
    pipeline.run(args.query_gene_set, args.background_gene_list, zip_output=args.zip_output)