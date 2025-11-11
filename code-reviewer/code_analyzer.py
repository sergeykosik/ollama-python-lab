"""
Code Analyzer Module
Handles code analysis using Ollama LLMs with context awareness
"""

import requests
import json
from typing import List, Dict, Any
import re


class CodeAnalyzer:
    def __init__(self, ollama_host: str = "http://localhost:11434", model: str = "codellama"):
        self.ollama_host = ollama_host
        self.model = model
        self.api_url = f"{ollama_host}/api/generate"

    def update_settings(self, ollama_host: str, model: str):
        """Update Ollama connection settings"""
        self.ollama_host = ollama_host
        self.model = model
        self.api_url = f"{ollama_host}/api/generate"

    def analyze(self, files_data: List[Dict], context_files: List[Dict],
                config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive code analysis with context

        Args:
            files_data: List of files to review with name, content, language
            context_files: List of context files from codebase
            config: Review configuration (depth, focus_areas, etc.)

        Returns:
            Dictionary with review results
        """
        results = {
            'total_issues': 0,
            'critical_count': 0,
            'warning_count': 0,
            'suggestion_count': 0,
            'file_reviews': [],
            'overall_feedback': ''
        }

        # Analyze each file
        for file_data in files_data:
            file_review = self._analyze_file(file_data, context_files, config)
            results['file_reviews'].append(file_review)

            # Aggregate counts
            for issue in file_review.get('issues', []):
                results['total_issues'] += 1
                severity = issue.get('severity', 'info')
                if severity == 'critical':
                    results['critical_count'] += 1
                elif severity == 'warning':
                    results['warning_count'] += 1
                elif severity == 'suggestion':
                    results['suggestion_count'] += 1

        # Generate overall feedback
        results['overall_feedback'] = self._generate_overall_feedback(
            files_data, results['file_reviews'], config
        )

        return results

    def _analyze_file(self, file_data: Dict, context_files: List[Dict],
                      config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single file with context"""

        # Build analysis prompt
        prompt = self._build_analysis_prompt(file_data, context_files, config)

        # Call Ollama
        try:
            response = self._call_ollama(prompt)
            parsed_review = self._parse_review_response(response, file_data)
        except Exception as e:
            parsed_review = {
                'filename': file_data['name'],
                'language': file_data.get('language', 'unknown'),
                'summary': f'Error during analysis: {str(e)}',
                'issues': [],
                'positive_aspects': []
            }

        return parsed_review

    def _build_analysis_prompt(self, file_data: Dict, context_files: List[Dict],
                                config: Dict[str, Any]) -> str:
        """Build the analysis prompt for the LLM"""

        depth_instructions = {
            'Quick': 'Provide a quick overview of major issues only.',
            'Standard': 'Provide a balanced review covering main issues and improvements.',
            'Deep': 'Perform a thorough analysis covering all aspects in detail.',
            'Comprehensive': 'Perform an exhaustive analysis including edge cases, architectural patterns, and subtle issues.'
        }

        focus_description = ', '.join(config.get('focus_areas', []))

        prompt = f"""You are an expert code reviewer. Analyze the following code file and provide a comprehensive review.

REVIEW CONFIGURATION:
- Review Depth: {config.get('depth', 'Standard')}
- Focus Areas: {focus_description}
- Instructions: {depth_instructions.get(config.get('depth', 'Standard'), '')}

FILE TO REVIEW:
Filename: {file_data['name']}
Language: {file_data.get('language', 'unknown')}

```{file_data.get('language', '')}
{file_data['content']}
```
"""

        # Add context if available
        if context_files:
            prompt += f"\n\nCODEBASE CONTEXT ({len(context_files)} related files):\n"
            for idx, ctx_file in enumerate(context_files[:3], 1):  # Limit to top 3 for token efficiency
                prompt += f"\n--- Context File {idx}: {ctx_file.get('filename', 'unknown')} ---\n"
                # Include snippet of context file
                content = ctx_file.get('content', '')
                snippet = content[:500] + '...' if len(content) > 500 else content
                prompt += f"```\n{snippet}\n```\n"

        prompt += """

REVIEW REQUIREMENTS:

Provide your review in the following structured format:

## SUMMARY
[Provide a brief 2-3 sentence summary of the code and its overall quality]

## ISSUES
[List each issue in the following format:]

ISSUE: [Type of issue]
SEVERITY: [critical/warning/suggestion]
LINE: [line number if applicable, or 'general']
MESSAGE: [Detailed description of the issue]
SUGGESTION: [How to fix this issue]
"""

        if config.get('include_examples'):
            prompt += "CODE_EXAMPLE: [Show corrected code example if applicable]\n"

        prompt += """

## POSITIVE ASPECTS
[List positive aspects of the code, good practices observed]

Focus on being specific, actionable, and constructive in your feedback.
"""

        return prompt

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API for code analysis"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Lower temperature for more consistent analysis
                "top_p": 0.9,
                "top_k": 40
            }
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120  # 2 minute timeout for complex analysis
            )
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API error: {str(e)}")

    def _parse_review_response(self, response: str, file_data: Dict) -> Dict[str, Any]:
        """Parse the LLM response into structured format"""

        review = {
            'filename': file_data['name'],
            'language': file_data.get('language', 'unknown'),
            'summary': '',
            'issues': [],
            'positive_aspects': []
        }

        # Extract summary
        summary_match = re.search(r'## SUMMARY\s*(.*?)(?=##|\Z)', response, re.DOTALL | re.IGNORECASE)
        if summary_match:
            review['summary'] = summary_match.group(1).strip()

        # Extract issues
        issues_section = re.search(r'## ISSUES\s*(.*?)(?=## POSITIVE|##\s*$|\Z)', response, re.DOTALL | re.IGNORECASE)
        if issues_section:
            issues_text = issues_section.group(1)
            issue_blocks = re.split(r'\n(?=ISSUE:)', issues_text)

            for block in issue_blocks:
                if not block.strip():
                    continue

                issue = {}

                # Extract issue type
                type_match = re.search(r'ISSUE:\s*(.+?)(?=\n|$)', block, re.IGNORECASE)
                if type_match:
                    issue['type'] = type_match.group(1).strip()

                # Extract severity
                severity_match = re.search(r'SEVERITY:\s*(\w+)', block, re.IGNORECASE)
                if severity_match:
                    issue['severity'] = severity_match.group(1).strip().lower()
                else:
                    issue['severity'] = 'info'

                # Extract line number
                line_match = re.search(r'LINE:\s*(.+?)(?=\n|$)', block, re.IGNORECASE)
                if line_match:
                    line_str = line_match.group(1).strip()
                    try:
                        issue['line'] = int(line_str)
                    except ValueError:
                        issue['line'] = line_str

                # Extract message
                message_match = re.search(r'MESSAGE:\s*(.+?)(?=SUGGESTION:|CODE_EXAMPLE:|\n(?:ISSUE|SEVERITY|LINE|MESSAGE):|\Z)',
                                          block, re.DOTALL | re.IGNORECASE)
                if message_match:
                    issue['message'] = message_match.group(1).strip()

                # Extract suggestion
                suggestion_match = re.search(r'SUGGESTION:\s*(.+?)(?=CODE_EXAMPLE:|\n(?:ISSUE|SEVERITY|LINE|MESSAGE):|\Z)',
                                             block, re.DOTALL | re.IGNORECASE)
                if suggestion_match:
                    issue['suggestion'] = suggestion_match.group(1).strip()

                # Extract code example
                code_match = re.search(r'CODE_EXAMPLE:\s*(.+?)(?=\n(?:ISSUE|SEVERITY|LINE|MESSAGE):|\Z)',
                                       block, re.DOTALL | re.IGNORECASE)
                if code_match:
                    issue['code_example'] = code_match.group(1).strip()

                if issue.get('type') or issue.get('message'):
                    review['issues'].append(issue)

        # Extract positive aspects
        positive_section = re.search(r'## POSITIVE ASPECTS?\s*(.*?)(?=##|\Z)', response, re.DOTALL | re.IGNORECASE)
        if positive_section:
            positive_text = positive_section.group(1).strip()
            # Split by bullet points or newlines
            aspects = re.findall(r'[-•*]\s*(.+?)(?=\n[-•*]|\n\n|\Z)', positive_text, re.DOTALL)
            review['positive_aspects'] = [a.strip() for a in aspects if a.strip()]

        return review

    def _generate_overall_feedback(self, files_data: List[Dict],
                                    file_reviews: List[Dict], config: Dict[str, Any]) -> str:
        """Generate overall feedback across all reviewed files"""

        # Build summary prompt
        files_summary = "\n".join([f"- {f['name']}: {len(f['content'].splitlines())} lines"
                                   for f in files_data])

        total_issues = sum(len(r.get('issues', [])) for r in file_reviews)

        prompt = f"""Based on the code review of {len(files_data)} files, provide a brief overall assessment.

FILES REVIEWED:
{files_summary}

Total issues found: {total_issues}

Provide a 3-4 sentence overall assessment covering:
1. General code quality
2. Main strengths
3. Key areas for improvement
4. Overall recommendation

Keep it concise and actionable."""

        try:
            response = self._call_ollama(prompt)
            return response.strip()
        except Exception as e:
            return f"Overall: Reviewed {len(files_data)} files with {total_issues} issues identified. " \
                   f"See individual file reviews for details."
