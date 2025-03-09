# geneinsight/__init__.py
"""
GeneInsight: A package for gene analysis and interpretation.
"""
__version__ = "0.1.0"


# geneinsight/ontology/__init__.py
"""
Ontology-related functionality for gene analysis.

This package provides tools for:
1. Ontology enrichment analysis
2. Converting enrichment results to term-gene dictionaries
3. Orchestrating full ontology analysis workflows
"""
from geneinsight.ontology.workflow import OntologyWorkflow
from geneinsight.ontology.calculate_ontology_enrichment import (
    RAGModuleGSEAPY, 
    OntologyReader, 
    HypergeometricGSEA
)
from geneinsight.ontology.get_ontology_dictionary import (
    process_ontology_enrichment,
    save_ontology_dictionary
)

__all__ = [
    'OntologyWorkflow',
    'RAGModuleGSEAPY',
    'OntologyReader',
    'HypergeometricGSEA',
    'process_ontology_enrichment',
    'save_ontology_dictionary'
]


# geneinsight/ontology/ontology_folders/__init__.py
"""
Package containing default ontology files.

This package provides a set of default ontology files that can be used
with the ontology enrichment analysis workflow.
"""