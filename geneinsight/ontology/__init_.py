# geneinsight/ontology/__init__.py
"""
Ontology-related functionality for gene analysis.
This package provides tools for:
1. Ontology enrichment analysis
2. Converting enrichment results to term-gene dictionaries
3. Orchestrating full ontology analysis workflows
"""
# Import directly in the modules that need these, not here
# to avoid circular imports

__all__ = [
    'OntologyWorkflow',
    'RAGModuleGSEAPY',
    'OntologyReader',
    'HypergeometricGSEA',
    'process_ontology_enrichment',
    'save_ontology_dictionary'
]
"""
