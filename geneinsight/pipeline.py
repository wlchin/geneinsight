import os
import time
import logging
import tempfile
import shutil
from pathlib import Path
import pandas as pd
from typing import Dict, Optional, Tuple

# Rich for stage announcements
from rich.console import Console

logger = logging.getLogger(__name__)


class Pipeline:
    """Main pipeline for processing gene sets through topic modeling and enrichment."""

    def __init__(
        self,
        output_dir: str,
        temp_dir: Optional[str] = None,
        n_samples: int = 5,
        num_topics: Optional[int] = 10,
        pvalue_threshold: float = 0.01,
        api_service: str = "openai",
        api_model: str = "gpt-4o-mini",
        api_parallel_jobs: int = 4,
        api_base_url: Optional[str] = None,
        target_filtered_topics: int = 25,
        species: int = 9606,
        filtered_n_samples: int = 10,  # parameter for filtered sets topic modeling
        api_temperature: float = 0.2,   # temperature parameter
        call_ncbi_api: bool = True,     # whether to call NCBI API for gene summaries
        use_local_stringdb: bool = False  # whether to use local StringDB module instead of API
    ):
        self.output_dir = os.path.abspath(output_dir)

        # Rich console for step-by-step progress
        self.console = Console()

        # Use system temp directory if none specified
        if temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="topicgenes_")
            self._temp_is_system = True
        else:
            self.temp_dir = os.path.abspath(temp_dir)
            self._temp_is_system = False

        self.n_samples = n_samples
        self.num_topics = num_topics
        self.pvalue_threshold = pvalue_threshold
        self.api_service = api_service
        self.api_model = api_model
        self.api_parallel_jobs = api_parallel_jobs
        self.api_base_url = api_base_url
        self.target_filtered_topics = target_filtered_topics
        self.species = species  # Save species as an instance variable
        self.filtered_n_samples = filtered_n_samples  # store new parameter
        self.api_temperature = api_temperature  # store temperature parameter
        self.call_ncbi_api = call_ncbi_api  # store NCBI API control parameter
        self.use_local_stringdb = use_local_stringdb  # store local StringDB option

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
            "report": os.path.join(self.temp_dir, "report"),
        }

        # Create all subdirectories
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        # Create sphinx_builds directory as a subfolder in the results folder.
        self.dirs["sphinx_builds"] = os.path.join(self.dirs["final"], "sphinx_builds")
        os.makedirs(self.dirs["sphinx_builds"], exist_ok=True)

        # Set timestamp for run
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")

        logger.debug(f"Pipeline initialized with output directory: {self.output_dir}")
        logger.debug(f"Using temporary directory: {self.temp_dir}")
        logger.debug(f"Species: {self.species}, NCBI API calls: {'enabled' if self.call_ncbi_api else 'disabled'}")
        logger.debug(f"StringDB mode: {'local' if self.use_local_stringdb else 'API'}")

    def run(
        self,
        query_gene_set: str,
        background_gene_list: str,
        generate_report: bool = True,
        report_title: Optional[str] = None,
    ) -> str:
        """
        Run the full pipeline.
        """
        self.console.rule("[bold green]Starting Pipeline[/bold green]")
        self.console.print(f"[bold]Query Gene Set:[/bold] {query_gene_set}")
        self.console.print(f"[bold]Background Gene List:[/bold] {background_gene_list}")
        logger.info(f"Starting pipeline run with query gene set: {query_gene_set}")

        # Extract gene set name from file path
        gene_set_name = os.path.splitext(os.path.basename(query_gene_set))[0]
        run_id = f"{gene_set_name}_{self.timestamp}"
        logger.info(f"Run ID: {run_id}")

        # Check for overlap between query and background
        try:
            self.console.print("[bold]Checking overlap between query and background...[/bold]")
            query_genes = set(pd.read_csv(query_gene_set, header=None)[0].unique())
            background_genes = set(pd.read_csv(background_gene_list, header=None)[0].unique())
        except Exception as e:
            self.console.print("No common items between query gene set and background. Aborting pipeline.")
            self.console.print(f"[bold red]ERROR: Failed to read gene files: {e}[/bold red]")
            logger.error(f"Failed to read gene files: {e}")
            raise

        overlap = query_genes.intersection(background_genes)
        if not overlap:
            self.console.print("No common items between query gene set and background. Aborting pipeline.")
            raise ValueError("No overlap found between query gene set and background. Pipeline aborted.")
        else:
            self.console.print(f"Found {len(overlap)} overlapping genes between query and background.")
            # New console print for Step 1 info
            self.console.print(f"[bold]Step 1 Info:[/bold] Query gene set length: {len(query_genes)}, Background gene set length: {len(background_genes)}, Species: {self.species}")
        try:
            # Step 1: Gene enrichment from StringDB
            self.console.rule("[bold cyan]Step 1: Retrieving gene enrichment data from StringDB[/bold cyan]")
            enrichment_df, documents_df = self._get_stringdb_enrichment(query_gene_set)
            
            # Check if enrichment or documents are empty
            if enrichment_df.empty or documents_df.empty:
                logger.error("No enrichment or documents returned from StringDB. Aborting pipeline.")
                self.console.print("[bold red]ERROR: No enrichment or documents returned from StringDB. Aborting pipeline.[/bold red]")
                raise ValueError("No enrichment or documents returned from StringDB. Pipeline aborted.")
                
            self.console.print(f"Retrieved {len(enrichment_df)} enriched terms and {len(documents_df)} documents from StringDB.")
            self.console.print("[bold green]Done! Step 1 completed.[/bold green]")

            # Step 2: Topic modeling
            self.console.rule("[bold cyan]Step 2: Running topic modeling[/bold cyan]")
            self.console.print(f"Running topic modeling with {self.n_samples} rounds of sampling on {len(documents_df)} STRING-DB terms")
            topics_df = self._run_topic_modeling(documents_df)
            self.console.print("[bold green]Done! Step 2 completed.[/bold green]")

            # Step 3: Prompt generation
            self.console.rule("[bold cyan]Step 3: Generating prompts for topic refinement[/bold cyan]")
            self.console.print("Generating prompts for API calls")
            prompts_df = self._generate_prompts(topics_df)
            self.console.print("[bold green]Done! Step 3 completed.[/bold green]")

            # Step 4: API processing
            self.console.rule("[bold cyan]Step 4: Processing prompts through API[/bold cyan]")
            self.console.print(f"[bold]Using API: {self.api_service} with {self.api_parallel_jobs} parallel jobs and temperature: {self.api_temperature}[/bold]")
            api_results_df = self._process_api_calls(prompts_df)
            self.console.print("[bold green]Done! Step 4 completed.[/bold green]")

            # Step 5: Create summary
            self.console.rule("[bold cyan]Step 5: Creating summary[/bold cyan]")
            self.console.print("Creating summary by combining API results with enrichment data")
            summary_df = self._create_summary(api_results_df, enrichment_df)
            self.console.print("[bold green]Done! Step 5 completed.[/bold green]")

            # Step 6: Hypergeometric enrichment
            enriched_df = self._perform_hypergeometric_enrichment(
                summary_df, query_gene_set, background_gene_list
            )
            if enriched_df.empty:
                logger.error("Hypergeometric enrichment resulted in an empty DataFrame")
                raise ValueError("Hypergeometric enrichment resulted in an empty DataFrame")
            self.console.print(f"Number of enriched gene sets: {len(enriched_df)}")

            # Step 7: Topic modeling on filtered gene sets (meta-analysis)
            self.console.rule("[bold cyan]Step 7: Running topic modeling on filtered gene sets[/bold cyan]")
            self.console.print(f"Running topic modeling on filtered gene sets using {self.filtered_n_samples} samples")
            topics_df = self._run_topic_modeling_on_filtered_sets(enriched_df)
            self.console.print("[bold green]Done! Step 7 completed.[/bold green]")

            # Step 8: Extract key topics
            self.console.rule("[bold cyan]Step 8: Extracting key topics[/bold cyan]")
            self.console.print("Detecting key topics (centroids) from the enriched gene sets")
            key_topics_df = self._get_key_topics(topics_df)
            self.console.print("[bold green]Done! Step 8 completed.[/bold green]")

            # Step 9: Filter topics by similarity
            self.console.rule("[bold cyan]Step 9: Filtering topics by similarity[/bold cyan]")
            self.console.print("Filtering topics for report generation")
            self.console.print("Number of key topics before filtering: ", len(key_topics_df))
            filtered_df = self._filter_topics(key_topics_df)
            self.console.print(f"Number of topics filtered for report: {len(filtered_df)}")
            self.console.print("[bold green]Done! Step 9 completed.[/bold green]")

            if len(filtered_df) < self.target_filtered_topics:
                logger.info("Filtered topics fewer than target; using entire enriched dataframe for filtering and clustering.")
                filtered_df = self._filter_topics(enriched_df)

            # Step 9b: Clustering filtered topics
            self.console.rule("[bold cyan]Step 9b: Clustering filtered topics[/bold cyan]")
            self.console.print("Creating hierarchical summaries")
            clustered_df = self._run_clustering(filtered_df)
            self.console.print("[bold green]Done! Step 9b completed.[/bold green]")

            # Step 9c: Ontology enrichment analysis
            self.console.rule("[bold cyan]Step 9c: Performing ontology enrichment analysis[/bold cyan]")
            self.console.print("Running ontology enrichment analysis against GO and HPO ontologies")
            ontology_dict_df = self._perform_ontology_enrichment(
                summary_df=summary_df,
                clustered_df=clustered_df,
                query_gene_set=query_gene_set,
                background_gene_list=background_gene_list
            )
            self.console.print("[bold green]Done! Step 9c completed.[/bold green]")

            # Step 10: Finalize outputs
            self.console.rule("[bold cyan]Step 10: Finalizing outputs[/bold cyan]")
            self.console.print("Generating report metadata and checking output files ...")
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
                    "ontology_dict": ontology_dict_df,
                }
            )
            self.console.print("[bold green]Done! Step 10 completed.[/bold green]")

            # Step 11: Generate report (if requested)
            if generate_report:
                self.console.rule("[bold cyan]Step 11: Generating report[/bold cyan]")
                report_output = self._generate_report(
                    output_path=output_path,
                    query_gene_set=query_gene_set,
                    report_title=report_title
                )
                if report_output:
                    logger.info(f"Report generated successfully at {report_output}")
                else:
                    logger.warning("Report generation failed or was skipped")
                self.console.print("[bold green]Done! Step 11 completed.[/bold green]")

            # Step 12: Reorganize output directory
            self.console.rule("[bold cyan]Step 12: Reorganizing output directory[/bold cyan]")
            if os.path.isdir(output_path):
                self._reorganize_output_directory(output_path)
            self.console.print("[bold green]Done! Step 12 completed.[/bold green]")

            self.console.print("[bold green]Pipeline completed successfully![/bold green]")
            self.console.print(f"Results available at: [bold]{output_path + '.zip'}[/bold]")
            if generate_report:
                self.console.print(f"Report available at: [bold]{self.dirs['sphinx_builds']+ '.zip'}[/bold]")
            logger.info(f"Pipeline completed successfully. Results available at: {self.dirs['final']}")
            return self.dirs["final"]

        except Exception as e:
            logger.error(f"Error in pipeline: {e}", exc_info=True)
            raise
        finally:
            if self._temp_is_system:
                self._cleanup_temp()

    def _cleanup_temp(self):
        """Clean up temporary directory if it's system-generated."""
        if hasattr(self, '_temp_is_system') and self._temp_is_system and os.path.exists(self.temp_dir):
            try:
                logger.debug(f"Cleaning up temporary directory: {self.temp_dir}")
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up temporary directory: {e}")

    def _reorganize_output_directory(self, output_path: str):
        """Reorganize the output directory structure at the results level."""
        try:
            results_dir = self.dirs["final"]
            logger.debug(f"Reorganizing output directory structure in: {results_dir}")
            
            # Get the run directory (where API calls, clustering, etc. outputs are stored)
            # This should be the output_path parameter which is the run_dir from _finalize_outputs
            run_dir = output_path
            if not os.path.isdir(run_dir):
                logger.warning(f"Run directory not found: {run_dir}")
                run_dir = results_dir
            
            # Handle ontology folder - move contents to the run directory
            ontology_path = os.path.join(results_dir, "ontology")
            if os.path.exists(ontology_path):
                logger.debug(f"Moving contents of ontology folder to run directory: {ontology_path} -> {run_dir}")
                # List all files and directories in the ontology folder
                ontology_contents = os.listdir(ontology_path)
                for item in ontology_contents:
                    src_path = os.path.join(ontology_path, item)
                    dst_path = os.path.join(run_dir, item)
                    
                    # Handle name conflicts by adding a suffix if needed
                    if os.path.exists(dst_path):
                        base, ext = os.path.splitext(item)
                        dst_path = os.path.join(run_dir, f"{base}_ontology{ext}")
                        logger.debug(f"Renaming {item} to {os.path.basename(dst_path)} to avoid conflicts")
                    
                    # Move the item to run directory
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path)
                    else:
                        shutil.copy2(src_path, dst_path)
                
                # Remove the original ontology folder after moving contents
                logger.debug(f"Removing original ontology folder after moving contents: {ontology_path}")
                shutil.rmtree(ontology_path)
            
            # Handle sphinx builds directory
            sphinx_source_path = os.path.join(results_dir, "sphinx_builds")
            if os.path.exists(sphinx_source_path):
                sphinx_nested_builds = os.path.join(sphinx_source_path, "results", "sphinx_builds")
                if os.path.exists(sphinx_nested_builds):
                    sphinx_temp_path = os.path.join(results_dir, "sphinx_builds_temp")
                    logger.debug(f"Copying sphinx builds to top level: {sphinx_nested_builds} -> {sphinx_temp_path}")
                    shutil.copytree(sphinx_nested_builds, sphinx_temp_path)
                    logger.debug(f"Removing original sphinx_builds folder: {sphinx_source_path}")
                    shutil.rmtree(sphinx_source_path)
                    logger.debug("Renaming temporary sphinx_builds folder to final location")
                    os.rename(sphinx_temp_path, sphinx_source_path)
                else:
                    logger.warning(f"Nested sphinx_builds directory not found: {sphinx_nested_builds}")
            else:
                logger.warning(f"Sphinx builds directory not found: {sphinx_source_path}")
            
            self._zip_results_folders(results_dir)
            logger.debug("Output directory reorganization completed")
        
        except Exception as e:
            logger.error(f"Error reorganizing output directory: {e}", exc_info=True)

    def _zip_results_folders(self, results_dir: str):
        """Zip each folder within the results directory."""
        from .utils.zip_helper import zip_directory
        try:
            logger.debug("Zipping folders in results directory")
            dirs_to_zip = [
                d for d in os.listdir(results_dir)
                if os.path.isdir(os.path.join(results_dir, d))
            ]
            for dir_name in dirs_to_zip:
                dir_path = os.path.join(results_dir, dir_name)
                zip_path = os.path.join(results_dir, f"{dir_name}.zip")
                logger.debug(f"Zipping directory: {dir_path} -> {zip_path}")
                zip_directory(dir_path, zip_path)
                shutil.rmtree(dir_path)
            logger.debug(f"Zipped {len(dirs_to_zip)} directories in results folder")
        except Exception as e:
            logger.error(f"Error zipping results folders: {e}", exc_info=True)

    def _get_stringdb_enrichment(self, query_gene_set: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get gene enrichment data from StringDB using either API or local module."""
        if self.species in [9606, 10090] and self.use_local_stringdb:
            from .enrichment.stringdb_local import process_gene_enrichment
            self.console.print("[bold]Using local StringDB module for gene enrichment (9606 or 10090)[/bold]")
            
            enrichment_output = os.path.join(self.dirs["enrichment"], "enrichment.csv")
            documents_output = os.path.join(self.dirs["enrichment"], "documents.csv")
            
            enrichment_df, documents = process_gene_enrichment(
                input_file=query_gene_set,
                output_dir=self.dirs["enrichment"],
                species=self.species
            )
        else:
            from .enrichment.stringdb import process_gene_enrichment
            self.console.print("[bold]Using StringDB API for gene enrichment[/bold]")
            
            enrichment_output = os.path.join(self.dirs["enrichment"], "enrichment.csv")
            documents_output = os.path.join(self.dirs["enrichment"], "documents.csv")
            
            enrichment_df, documents = process_gene_enrichment(
                input_file=query_gene_set,
                output_dir=self.dirs["enrichment"],
                species=self.species,
                mode="single"
            )
            
        return enrichment_df, pd.DataFrame({"description": documents})

    def _run_topic_modeling(self, documents_df: pd.DataFrame) -> pd.DataFrame:
        """Run topic modeling on documents."""
        from .models.bertopic import run_multiple_seed_topic_modeling
        documents_path = os.path.join(self.dirs["enrichment"], "documents_for_modeling.csv")
        topics_output = os.path.join(self.dirs["topics"], "topics.csv")
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
        if not os.path.exists(prompts_path):
            prompts_df.to_csv(prompts_path, index=False)
        logger.info("Running API calls without caching.")
        api_results_df = batch_process_api_calls(
            prompts_csv=prompts_path,
            output_api=api_output,
            service=self.api_service,
            model=self.api_model,
            base_url=self.api_base_url,
            n_jobs=self.api_parallel_jobs,
            temperature=self.api_temperature  # pass temperature parameter
        )
        return api_results_df

    def _create_summary(self, api_results_df: pd.DataFrame, enrichment_df: pd.DataFrame) -> pd.DataFrame:
        """Create summary by combining API results with enrichment data."""
        from .analysis.summary import create_summary
        api_path = os.path.join(self.dirs["minor_topics"], "api_results.csv")
        enrichment_path = os.path.join(self.dirs["enrichment"], "enrichment.csv")
        summary_output = os.path.join(self.dirs["summary"], "summary.csv")
        if not os.path.exists(api_path):
            api_results_df.to_csv(api_path, index=False)
        if not os.path.exists(enrichment_path):
            enrichment_df.to_csv(enrichment_path, index=False)
        summary_df = create_summary(api_results_df, enrichment_df, summary_output)
        return summary_df

    def _perform_hypergeometric_enrichment(self, summary_df: pd.DataFrame,
                                           query_gene_set: str,
                                           background_gene_list: str) -> pd.DataFrame:
        """Perform hypergeometric enrichment analysis."""
        from .enrichment.hypergeometric import hypergeometric_enrichment
        summary_path = os.path.join(self.dirs["summary"], "summary.csv")
        enriched_output = os.path.join(self.dirs["filtered_sets"], "enriched.csv")
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
            n_samples=self.filtered_n_samples
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
            top_n=None
        )
        logger.info(f"Found {len(key_topics_df)} key topics")
        return key_topics_df

    def _filter_topics(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Filter topics by similarity."""
        from .analysis.similarity import filter_terms_by_similarity
        
        # Check if input DataFrame is empty
        if input_df.empty:
            logger.error("Cannot filter topics: input DataFrame is empty")
            raise ValueError("Cannot filter topics: input DataFrame is empty")
            
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
        """Run clustering on the filtered topics."""
        from .analysis.clustering import run_clustering
        clustering_input = os.path.join(self.dirs["filtered_sets"], "filtered.csv")
        filtered_df.to_csv(clustering_input, index=False)
        clustering_output = os.path.join(self.dirs["filtered_sets"], "clustered.csv")
        run_clustering(
            input_csv=clustering_input,
            output_csv=clustering_output,
            min_clusters=5,
            max_clusters=10,
            n_trials=100
        )
        clustered_df = pd.read_csv(clustering_output)
        return clustered_df

    def _generate_report(self, output_path: str, query_gene_set: str,
                         report_title: Optional[str] = None) -> Optional[str]:
        """Generate an HTML report using geneinsight.reports.pipeline.run_pipeline."""
        logger.info("Generating HTML report using geneinsight.reports.pipeline.run_pipeline...")
        gene_set = os.path.splitext(os.path.basename(query_gene_set))[0]
        report_out_dir = self.dirs["sphinx_builds"]
        os.makedirs(report_out_dir, exist_ok=True)

        try:
            from .report import pipeline as reports_pipeline
        except ImportError as e:
            logger.error(f"Failed to import geneinsight.reports.pipeline: {e}")
            return None

        # Convert numeric species ID to string taxonomy ID format for NCBI
        taxonomy_id = str(self.species)
        logger.info(f"Generating report for taxonomy ID: {taxonomy_id}")
        logger.info(f"NCBI API calls: {'enabled' if self.call_ncbi_api else 'disabled'}")

        status, html_index = reports_pipeline.run_pipeline(
            input_folder=output_path,
            output_folder=report_out_dir,
            gene_set=gene_set,
            context_service=self.api_service,
            context_api_key=None,
            context_model=self.api_model,
            context_base_url=self.api_base_url,
            taxonomy_id=taxonomy_id,      # Pass the taxonomy ID to the report pipeline
            call_ncbi_api=self.call_ncbi_api  # Use the class parameter for NCBI API calls
        )
        if status and html_index:
            logger.info(f"Report generated successfully at {html_index}")
            return html_index
        else:
            logger.error("Report generation failed.")
            return None

    def _finalize_outputs(self, run_id: str, dataframes: Dict[str, pd.DataFrame]) -> str:
        """
        Finalize outputs by copying to the output directory.
        """
        run_dir = os.path.join(self.dirs["final"], run_id)
        os.makedirs(run_dir, exist_ok=True)

        for name, df in dataframes.items():
            output_path = os.path.join(run_dir, f"{name}.csv")
            df.to_csv(output_path, index=False)
            logger.debug(f"Saved {name} to {output_path}")

        metadata = {
            "run_id": run_id,
            "timestamp": self.timestamp,
            "n_samples": self.n_samples,
            "num_topics": self.num_topics,
            "pvalue_threshold": self.pvalue_threshold,
            "api_service": self.api_service,
            "api_model": self.api_model,
            "species": self.species,
            "call_ncbi_api": self.call_ncbi_api,  # Include NCBI API setting in metadata
            "use_local_stringdb": self.use_local_stringdb,  # Include local StringDB setting in metadata
        }

        metadata_path = os.path.join(run_dir, "metadata.csv")
        pd.DataFrame([metadata]).to_csv(metadata_path, index=False)

        return run_dir

    def _perform_ontology_enrichment(self, summary_df: pd.DataFrame,
                                        clustered_df: pd.DataFrame,
                                        query_gene_set: str,
                                        background_gene_list: str) -> pd.DataFrame:
        """
        Perform ontology enrichment analysis using the ontology workflow.
        """
        logger.info("Running ontology enrichment analysis")
        ontology_folder = os.path.join(os.path.dirname(__file__), "ontology", "ontology_folders")
        from .ontology.workflow import OntologyWorkflow
        workflow = OntologyWorkflow(
            ontology_folder=ontology_folder,
            fdr_threshold=self.pvalue_threshold,
            use_temp_files=False
        )
        ontology_output_dir = os.path.join(self.dirs["final"], "ontology")
        os.makedirs(ontology_output_dir, exist_ok=True)
        enrichment_df, ontology_dict_df = workflow.process_dataframes(
            summary_df=summary_df,
            clustered_df=clustered_df,
            gene_list_path=query_gene_set,
            background_genes_path=background_gene_list,
            output_dir=ontology_output_dir
        )
        logger.info(f"Ontology enrichment complete. Found {len(ontology_dict_df)} ontology dictionaries.")
        return ontology_dict_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the TopicGenes pipeline.")
    parser.add_argument("--species", type=int, default=9606, help="Species identifier (default: 9606 for human).")
    parser.add_argument("query_gene_set", help="Path to file containing query gene set.")
    parser.add_argument("background_gene_list", help="Path to file containing background gene list.")
    parser.add_argument("-o", "--output_dir", default="./output", help="Directory to store final outputs.")
    parser.add_argument("--no-report", action="store_true", help="Skip generating an HTML report.")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of topic models to run with different seeds.")
    parser.add_argument("--num_topics", type=int, default=None, help="Number of topics to extract in topic modeling.")
    parser.add_argument("--pvalue_threshold", type=float, default=0.05, help="Adjusted P-value threshold.")
    parser.add_argument("--api_service", type=str, default="openai", help="API service for topic refinement.")
    parser.add_argument("--api_model", type=str, default="gpt-4o-mini", help="Model name for the API service.")
    parser.add_argument("--api_parallel_jobs", type=int, default=4, help="Number of parallel API jobs.")
    parser.add_argument("--api_base_url", type=str, default=None, help="Base URL for the API service, if needed.")
    parser.add_argument("--target_filtered_topics", type=int, default=25, help="Target number of topics.")
    parser.add_argument("--temp_dir", type=str, default=None, help="Temporary directory for intermediate files.")
    parser.add_argument("--report_title", type=str, default=None, help="Title for the generated report.")
    parser.add_argument("--filtered_n_samples", type=int, default=10, help="Number of topic models to run on filtered sets.")
    parser.add_argument("--api_temperature", type=float, default=0.2,
                        help="Sampling temperature for API calls (default: 0.2).")
    parser.add_argument("--no-ncbi-api", action="store_true", help="Disable NCBI API calls for gene summaries.")
    parser.add_argument("--use-local-stringdb", action="store_true", help="Use local StringDB module instead of API.")
    parser.add_argument("-v", "--verbosity",
                        default="info",
                        choices=["none", "debug", "info", "warning", "error", "critical"],
                        help="Set logging verbosity (default: 'info'). Use 'none' to disable logging.")
    
    args = parser.parse_args()

    # Configure logging using RichHandler
    if args.verbosity == "none":
        logging.disable(logging.CRITICAL)
    else:
        level = getattr(logging, args.verbosity.upper(), logging.INFO)
        from rich.logging import RichHandler
        from rich.console import Console
        console = Console()
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=console, rich_tracebacks=True)]
        )

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
        species=args.species,
        filtered_n_samples=args.filtered_n_samples,
        api_temperature=args.api_temperature,
        call_ncbi_api=not args.no_ncbi_api,
        use_local_stringdb=args.use_local_stringdb  # Pass the local StringDB option
    )

    pipeline.run(
        query_gene_set=args.query_gene_set,
        background_gene_list=args.background_gene_list,
        generate_report=not args.no_report,
        report_title=args.report_title
    )