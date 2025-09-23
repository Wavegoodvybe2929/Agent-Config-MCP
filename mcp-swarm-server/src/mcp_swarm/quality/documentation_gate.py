"""
Documentation Quality Gate for MCP Swarm Intelligence Server

This module implements the documentation validation quality gate that ensures
comprehensive and accurate documentation coverage.
"""

from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import asyncio
import re
import ast
import logging
from dataclasses import dataclass
from enum import Enum

from mcp_swarm.quality.gate_engine import (
    BaseQualityGate, QualityGateResult, QualityGateType, QualityGateStatus
)

logger = logging.getLogger(__name__)


class DocumentationType(Enum):
    """Types of documentation"""
    DOCSTRING = "docstring"
    COMMENT = "comment"
    README = "readme"
    API_DOC = "api_doc"
    TUTORIAL = "tutorial"
    CHANGELOG = "changelog"


@dataclass
class DocumentationCoverage:
    """Documentation coverage metrics"""
    total_functions: int
    documented_functions: int
    total_classes: int
    documented_classes: int
    total_modules: int
    documented_modules: int
    coverage_percentage: float
    missing_docs: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_functions": self.total_functions,
            "documented_functions": self.documented_functions,
            "total_classes": self.total_classes,
            "documented_classes": self.documented_classes,
            "total_modules": self.total_modules,
            "documented_modules": self.documented_modules,
            "coverage_percentage": self.coverage_percentage,
            "missing_docs": self.missing_docs
        }


@dataclass
class DocumentationQuality:
    """Documentation quality metrics"""
    avg_docstring_length: float
    has_return_docs: bool
    has_parameter_docs: bool
    has_examples: bool
    has_type_hints: bool
    completeness_score: float
    quality_issues: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_docstring_length": self.avg_docstring_length,
            "has_return_docs": self.has_return_docs,
            "has_parameter_docs": self.has_parameter_docs,
            "has_examples": self.has_examples,
            "has_type_hints": self.has_type_hints,
            "completeness_score": self.completeness_score,
            "quality_issues": self.quality_issues
        }


@dataclass
class DocumentationResults:
    """Results from documentation analysis"""
    coverage: DocumentationCoverage
    quality: DocumentationQuality
    api_documentation: Dict[str, Any]
    file_documentation: Dict[str, Dict[str, Any]]
    overall_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "coverage": self.coverage.to_dict(),
            "quality": self.quality.to_dict(),
            "api_documentation": self.api_documentation,
            "file_documentation": self.file_documentation,
            "overall_score": self.overall_score
        }


class PythonCodeAnalyzer:
    """Analyzer for Python code documentation"""
    
    def __init__(self):
        self.function_pattern = re.compile(r'^\s*def\s+(\w+)', re.MULTILINE)
        self.class_pattern = re.compile(r'^\s*class\s+(\w+)', re.MULTILINE)
        self.docstring_pattern = re.compile(r'"""(.*?)"""', re.DOTALL)
        
    async def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file for documentation"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST for detailed analysis
            try:
                tree = ast.parse(content)
                ast_analysis = self._analyze_ast(tree)
            except SyntaxError as e:
                logger.warning("Syntax error in %s: %s", file_path, e)
                ast_analysis = {
                    "functions": [],
                    "classes": [],
                    "module_docstring": None,
                    "imports": []
                }
            
            # Basic regex analysis as fallback
            functions = self.function_pattern.findall(content)
            classes = self.class_pattern.findall(content)
            docstrings = self.docstring_pattern.findall(content)
            
            # Calculate coverage
            total_items = len(functions) + len(classes) + 1  # +1 for module
            documented_functions = len([f for f in ast_analysis["functions"] if f.get("docstring")])
            documented_classes = len([c for c in ast_analysis["classes"] if c.get("docstring")])
            has_module_doc = ast_analysis["module_docstring"] is not None
            
            documented_items = documented_functions + documented_classes + (1 if has_module_doc else 0)
            coverage_percent = (documented_items / total_items * 100) if total_items > 0 else 0
            
            # Quality analysis
            quality_metrics = self._analyze_documentation_quality(ast_analysis, content)
            
            return {
                "file_path": str(file_path),
                "functions": functions,
                "classes": classes,
                "docstrings": docstrings,
                "ast_analysis": ast_analysis,
                "coverage_percent": coverage_percent,
                "documented_functions": documented_functions,
                "documented_classes": documented_classes,
                "has_module_doc": has_module_doc,
                "quality_metrics": quality_metrics
            }
            
        except (OSError, UnicodeDecodeError) as e:
            logger.error("Error analyzing file %s: %s", file_path, e)
            return {
                "file_path": str(file_path),
                "error": str(e),
                "coverage_percent": 0,
                "documented_functions": 0,
                "documented_classes": 0,
                "has_module_doc": False
            }
    
    def _analyze_ast(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze AST for detailed documentation information"""
        
        functions = []
        classes = []
        module_docstring = None
        imports = []
        
        # Get module docstring - check if tree is a Module
        if isinstance(tree, ast.Module):
            if (tree.body and isinstance(tree.body[0], ast.Expr) and
                isinstance(tree.body[0].value, ast.Constant) and
                isinstance(tree.body[0].value.value, str)):
                module_docstring = tree.body[0].value.value
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "docstring": ast.get_docstring(node),
                    "args": [arg.arg for arg in node.args.args],
                    "returns": node.returns is not None,
                    "decorators": [ast.unparse(d) if hasattr(ast, 'unparse') else str(d) for d in node.decorator_list],
                    "line_number": node.lineno
                }
                functions.append(func_info)
            
            elif isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "docstring": ast.get_docstring(node),
                    "methods": [],
                    "bases": [ast.unparse(base) if hasattr(ast, 'unparse') else str(base) for base in node.bases],
                    "decorators": [ast.unparse(d) if hasattr(ast, 'unparse') else str(d) for d in node.decorator_list],
                    "line_number": node.lineno
                }
                
                # Analyze methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_info = {
                            "name": item.name,
                            "docstring": ast.get_docstring(item),
                            "args": [arg.arg for arg in item.args.args],
                            "returns": item.returns is not None,
                            "line_number": item.lineno
                        }
                        class_info["methods"].append(method_info)
                
                classes.append(class_info)
            
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append(name.name)
                else:
                    imports.append(node.module or "")
        
        return {
            "functions": functions,
            "classes": classes,
            "module_docstring": module_docstring,
            "imports": imports
        }
    
    def _analyze_documentation_quality(self, ast_analysis: Dict[str, Any], content: str) -> Dict[str, Any]:
        """Analyze the quality of documentation"""
        
        docstrings = []
        
        # Collect all docstrings
        if ast_analysis["module_docstring"]:
            docstrings.append(ast_analysis["module_docstring"])
        
        for func in ast_analysis["functions"]:
            if func.get("docstring"):
                docstrings.append(func["docstring"])
        
        for cls in ast_analysis["classes"]:
            if cls.get("docstring"):
                docstrings.append(cls["docstring"])
            for method in cls.get("methods", []):
                if method.get("docstring"):
                    docstrings.append(method["docstring"])
        
        # Calculate quality metrics
        avg_length = sum(len(doc) for doc in docstrings) / len(docstrings) if docstrings else 0
        
        # Check for specific documentation patterns
        has_return_docs = any("return" in doc.lower() or "returns" in doc.lower() for doc in docstrings)
        has_parameter_docs = any("param" in doc.lower() or "arg" in doc.lower() or ":" in doc for doc in docstrings)
        has_examples = any("example" in doc.lower() or ">>>" in doc for doc in docstrings)
        has_type_hints = ":" in content and "->" in content  # Simple heuristic
        
        # Calculate completeness score
        completeness_factors = [
            avg_length > 20,  # Reasonable docstring length
            has_return_docs,
            has_parameter_docs,
            has_examples,
            has_type_hints
        ]
        completeness_score = sum(completeness_factors) / len(completeness_factors)
        
        # Identify quality issues
        quality_issues = []
        if avg_length < 20:
            quality_issues.append("Docstrings are too short on average")
        if not has_return_docs:
            quality_issues.append("Missing return documentation")
        if not has_parameter_docs:
            quality_issues.append("Missing parameter documentation")
        if not has_examples:
            quality_issues.append("Missing code examples")
        if not has_type_hints:
            quality_issues.append("Missing type hints")
        
        return {
            "avg_docstring_length": avg_length,
            "has_return_docs": has_return_docs,
            "has_parameter_docs": has_parameter_docs,
            "has_examples": has_examples,
            "has_type_hints": has_type_hints,
            "completeness_score": completeness_score,
            "quality_issues": quality_issues
        }


class DocumentationAnalyzer:
    """Main documentation analyzer"""
    
    def __init__(self):
        self.code_analyzer = PythonCodeAnalyzer()
        
    async def analyze_documentation(self, code_path: Path) -> DocumentationResults:
        """Analyze documentation across the entire codebase"""
        
        # Find all Python files
        python_files = list(code_path.rglob("*.py"))
        
        # Analyze each file
        file_results = {}
        for py_file in python_files:
            if self._should_analyze_file(py_file):
                file_results[str(py_file)] = await self.code_analyzer.analyze_file(py_file)
        
        # Calculate overall coverage
        coverage = self._calculate_coverage(file_results)
        
        # Calculate overall quality
        quality = self._calculate_quality(file_results)
        
        # Analyze API documentation
        api_docs = await self._analyze_api_documentation(code_path)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(coverage, quality, api_docs)
        
        return DocumentationResults(
            coverage=coverage,
            quality=quality,
            api_documentation=api_docs,
            file_documentation=file_results,
            overall_score=overall_score
        )
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Determine if a file should be analyzed"""
        
        # Skip test files, build files, and vendor directories
        skip_patterns = [
            "__pycache__",
            ".git",
            "build",
            "dist",
            ".venv",
            "venv",
            ".pytest_cache",
            "htmlcov"
        ]
        
        return not any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _calculate_coverage(self, file_results: Dict[str, Dict[str, Any]]) -> DocumentationCoverage:
        """Calculate overall documentation coverage"""
        
        total_functions = 0
        documented_functions = 0
        total_classes = 0
        documented_classes = 0
        total_modules = len(file_results)
        documented_modules = 0
        missing_docs = []
        
        for file_path, result in file_results.items():
            if "error" in result:
                continue
            
            # Functions
            if "functions" in result:
                total_functions += len(result["functions"])
                documented_functions += result.get("documented_functions", 0)
            
            # Classes  
            if "classes" in result:
                total_classes += len(result["classes"])
                documented_classes += result.get("documented_classes", 0)
            
            # Modules
            if result.get("has_module_doc", False):
                documented_modules += 1
            else:
                missing_docs.append(f"Module docstring missing in {file_path}")
            
            # Add specific missing documentation
            if "ast_analysis" in result:
                ast_data = result["ast_analysis"]
                
                for func in ast_data.get("functions", []):
                    if not func.get("docstring"):
                        missing_docs.append(f"Function {func['name']} missing docstring in {file_path}")
                
                for cls in ast_data.get("classes", []):
                    if not cls.get("docstring"):
                        missing_docs.append(f"Class {cls['name']} missing docstring in {file_path}")
        
        # Calculate coverage percentage
        total_items = total_functions + total_classes + total_modules
        documented_items = documented_functions + documented_classes + documented_modules
        coverage_percentage = (documented_items / total_items * 100) if total_items > 0 else 0
        
        return DocumentationCoverage(
            total_functions=total_functions,
            documented_functions=documented_functions,
            total_classes=total_classes,
            documented_classes=documented_classes,
            total_modules=total_modules,
            documented_modules=documented_modules,
            coverage_percentage=coverage_percentage,
            missing_docs=missing_docs[:50]  # Limit to first 50 for readability
        )
    
    def _calculate_quality(self, file_results: Dict[str, Dict[str, Any]]) -> DocumentationQuality:
        """Calculate overall documentation quality"""
        
        all_lengths = []
        return_docs_count = 0
        param_docs_count = 0
        examples_count = 0
        type_hints_count = 0
        total_files = 0
        all_quality_issues = []
        
        for result in file_results.values():
            if "error" in result or "quality_metrics" not in result:
                continue
            
            total_files += 1
            quality = result["quality_metrics"]
            
            all_lengths.append(quality.get("avg_docstring_length", 0))
            
            if quality.get("has_return_docs", False):
                return_docs_count += 1
            if quality.get("has_parameter_docs", False):
                param_docs_count += 1
            if quality.get("has_examples", False):
                examples_count += 1
            if quality.get("has_type_hints", False):
                type_hints_count += 1
            
            all_quality_issues.extend(quality.get("quality_issues", []))
        
        # Calculate averages
        avg_docstring_length = sum(all_lengths) / len(all_lengths) if all_lengths else 0
        has_return_docs = (return_docs_count / total_files) > 0.5 if total_files > 0 else False
        has_parameter_docs = (param_docs_count / total_files) > 0.5 if total_files > 0 else False
        has_examples = (examples_count / total_files) > 0.3 if total_files > 0 else False
        has_type_hints = (type_hints_count / total_files) > 0.7 if total_files > 0 else False
        
        # Calculate completeness score
        completeness_factors = [
            avg_docstring_length > 50,
            has_return_docs,
            has_parameter_docs,
            has_examples,
            has_type_hints
        ]
        completeness_score = sum(completeness_factors) / len(completeness_factors)
        
        # Get unique quality issues
        unique_issues = list(set(all_quality_issues))[:20]  # Limit to top 20
        
        return DocumentationQuality(
            avg_docstring_length=avg_docstring_length,
            has_return_docs=has_return_docs,
            has_parameter_docs=has_parameter_docs,
            has_examples=has_examples,
            has_type_hints=has_type_hints,
            completeness_score=completeness_score,
            quality_issues=unique_issues
        )
    
    async def _analyze_api_documentation(self, code_path: Path) -> Dict[str, Any]:
        """Analyze API documentation files"""
        
        api_docs = {
            "readme_exists": False,
            "api_docs_exist": False,
            "changelog_exists": False,
            "contributing_exists": False,
            "license_exists": False,
            "documentation_files": []
        }
        
        # Check for common documentation files
        doc_files = [
            ("README.md", "readme_exists"),
            ("README.rst", "readme_exists"),
            ("CHANGELOG.md", "changelog_exists"),
            ("CHANGELOG.rst", "changelog_exists"),
            ("CONTRIBUTING.md", "contributing_exists"),
            ("LICENSE", "license_exists"),
            ("LICENSE.md", "license_exists"),
            ("LICENSE.txt", "license_exists")
        ]
        
        for filename, key in doc_files:
            file_path = code_path / filename
            if file_path.exists():
                api_docs[key] = True
                api_docs["documentation_files"].append(filename)
        
        # Check for API documentation directory
        docs_dir = code_path / "docs"
        if docs_dir.exists() and docs_dir.is_dir():
            api_docs["api_docs_exist"] = True
            api_docs["documentation_files"].append("docs/")
        
        return api_docs
    
    def _calculate_overall_score(
        self, 
        coverage: DocumentationCoverage,
        quality: DocumentationQuality, 
        api_docs: Dict[str, Any]
    ) -> float:
        """Calculate overall documentation score"""
        
        # Coverage score (40% weight)
        coverage_score = coverage.coverage_percentage / 100
        
        # Quality score (40% weight) 
        quality_score = quality.completeness_score
        
        # API documentation score (20% weight)
        api_score = sum([
            api_docs.get("readme_exists", False),
            api_docs.get("api_docs_exist", False), 
            api_docs.get("changelog_exists", False),
            api_docs.get("contributing_exists", False),
            api_docs.get("license_exists", False)
        ]) / 5
        
        # Weighted average
        overall_score = (
            coverage_score * 0.4 +
            quality_score * 0.4 +
            api_score * 0.2
        )
        
        return overall_score


class DocumentationValidationGate(BaseQualityGate):
    """Quality gate for documentation validation and completeness"""
    
    def __init__(self, min_coverage: float = 80.0, min_quality: float = 0.7, timeout: float = 300.0):
        super().__init__("documentation_validation", timeout)
        self.min_coverage = min_coverage
        self.min_quality = min_quality
        self.analyzer = DocumentationAnalyzer()
        
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        """Execute documentation validation quality gate"""
        code_path = Path(context["code_path"])
        
        try:
            # Analyze documentation
            doc_results = await self.analyzer.analyze_documentation(code_path)
            
            # Generate recommendations
            recommendations = self._generate_documentation_recommendations(doc_results)
            
            # Determine status and score
            status = self._determine_status(doc_results)
            score = doc_results.overall_score
            
            # Prepare detailed results
            details = {
                "documentation_results": doc_results.to_dict(),
                "thresholds": {
                    "min_coverage": self.min_coverage,
                    "min_quality": self.min_quality
                },
                "summary": {
                    "total_items": (doc_results.coverage.total_functions + 
                                  doc_results.coverage.total_classes + 
                                  doc_results.coverage.total_modules),
                    "documented_items": (doc_results.coverage.documented_functions +
                                       doc_results.coverage.documented_classes +
                                       doc_results.coverage.documented_modules),
                    "missing_count": len(doc_results.coverage.missing_docs),
                    "quality_issues_count": len(doc_results.quality.quality_issues)
                }
            }
            
            return QualityGateResult(
                gate_type=QualityGateType.DOCUMENTATION_CHECK,
                status=status,
                score=score,
                details=details,
                recommendations=recommendations
            )
            
        except (ValueError, RuntimeError, OSError) as e:
            logger.error("Error executing documentation validation gate: %s", e)
            return QualityGateResult(
                gate_type=QualityGateType.DOCUMENTATION_CHECK,
                status=QualityGateStatus.FAILED,
                score=0.0,
                error_message=str(e),
                recommendations=["Check documentation validation configuration",
                               "Ensure all Python files are accessible",
                               "Verify file permissions and encoding"]
            )
    
    def _determine_status(self, results: DocumentationResults) -> QualityGateStatus:
        """Determine overall status based on documentation results"""
        
        coverage_percent = results.coverage.coverage_percentage
        quality_score = results.quality.completeness_score
        
        # Check minimum thresholds
        if coverage_percent < self.min_coverage * 0.7:  # Less than 70% of minimum
            return QualityGateStatus.FAILED
        
        if quality_score < self.min_quality * 0.7:  # Less than 70% of minimum quality
            return QualityGateStatus.FAILED
        
        # Check warning thresholds
        if coverage_percent < self.min_coverage or quality_score < self.min_quality:
            return QualityGateStatus.WARNING
        
        # Check overall score
        if results.overall_score < 0.6:
            return QualityGateStatus.WARNING
        
        return QualityGateStatus.PASSED
    
    def _generate_documentation_recommendations(self, results: DocumentationResults) -> List[str]:
        """Generate documentation improvement recommendations"""
        
        recommendations = []
        
        # Coverage recommendations
        coverage = results.coverage
        if coverage.coverage_percentage < self.min_coverage:
            recommendations.append(
                f"Increase documentation coverage from {coverage.coverage_percentage:.1f}% to {self.min_coverage}%"
            )
            
            if coverage.total_functions - coverage.documented_functions > 0:
                missing_funcs = coverage.total_functions - coverage.documented_functions
                recommendations.append(f"Add docstrings to {missing_funcs} functions")
            
            if coverage.total_classes - coverage.documented_classes > 0:
                missing_classes = coverage.total_classes - coverage.documented_classes
                recommendations.append(f"Add docstrings to {missing_classes} classes")
            
            if coverage.total_modules - coverage.documented_modules > 0:
                missing_modules = coverage.total_modules - coverage.documented_modules
                recommendations.append(f"Add module docstrings to {missing_modules} files")
        
        # Quality recommendations
        quality = results.quality
        if quality.completeness_score < self.min_quality:
            recommendations.append("Improve documentation quality")
            
            if not quality.has_return_docs:
                recommendations.append("Add return value documentation to functions")
            
            if not quality.has_parameter_docs:
                recommendations.append("Add parameter documentation to functions")
            
            if not quality.has_examples:
                recommendations.append("Add code examples to docstrings")
            
            if not quality.has_type_hints:
                recommendations.append("Add type hints to function signatures")
            
            if quality.avg_docstring_length < 50:
                recommendations.append("Write more detailed docstrings (current average too short)")
        
        # API documentation recommendations
        api_docs = results.api_documentation
        if not api_docs.get("readme_exists", False):
            recommendations.append("Create a README.md file")
        
        if not api_docs.get("api_docs_exist", False):
            recommendations.append("Create API documentation in docs/ directory")
        
        if not api_docs.get("changelog_exists", False):
            recommendations.append("Create a CHANGELOG.md file")
        
        if not api_docs.get("contributing_exists", False):
            recommendations.append("Create a CONTRIBUTING.md file")
        
        if not api_docs.get("license_exists", False):
            recommendations.append("Add a LICENSE file")
        
        # Specific missing documentation
        if results.coverage.missing_docs:
            recommendations.append("Address specific missing documentation:")
            for missing_doc in results.coverage.missing_docs[:5]:  # Show first 5
                recommendations.append(f"  • {missing_doc}")
            
            if len(results.coverage.missing_docs) > 5:
                remaining = len(results.coverage.missing_docs) - 5
                recommendations.append(f"  • ... and {remaining} more items")
        
        # Quality issues
        if results.quality.quality_issues:
            recommendations.append("Address documentation quality issues:")
            for issue in results.quality.quality_issues[:3]:  # Show first 3
                recommendations.append(f"  • {issue}")
        
        # Success recommendations
        if not recommendations:
            recommendations.extend([
                "Documentation coverage and quality are good",
                "Continue maintaining documentation standards",
                "Consider adding more code examples for better usability"
            ])
        
        return recommendations