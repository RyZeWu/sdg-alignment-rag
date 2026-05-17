import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import statistics
from collections import Counter

from langchain_text_splitters import TokenTextSplitter
from langchain.docstore.document import Document
import pandas as pd

logging.basicConfig(format="%(asctime)s - %(message)s", level="INFO")


class TokenAnalyzer:
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        # Initialize TokenTextSplitter with a large chunk size to count all tokens
        self.text_splitter = TokenTextSplitter(chunk_size=100000000, chunk_overlap=0)
        
    def count_tokens_in_text(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        try:
            # Use the TokenTextSplitter's internal tokenizer
            tokens = self.text_splitter._tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            logging.error(f"Error counting tokens: {e}")
            return 0
    
    def read_file(self, file_path: Path) -> str:
        """Read file content with proper encoding handling."""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logging.error(f"Error reading {file_path}: {e}")
                return ""
        
        logging.warning(f"Could not read {file_path} with any encoding")
        return ""
    
    def analyze_folder(self, extensions: List[str] = None) -> Dict:
        """
        Analyze all files in the folder and calculate token statistics.
        
        Args:
            extensions: List of file extensions to include (e.g., ['.txt', '.md'])
                       If None, includes all files.
        
        Returns:
            Dictionary with analysis results
        """
        logging.info(f"Analyzing folder: {self.folder_path}")
        
        file_token_counts = {}
        all_token_counts = []
        
        # Get all files in the folder
        files = []
        for root, dirs, filenames in os.walk(self.folder_path):
            for filename in filenames:
                file_path = Path(root) / filename
                
                # Filter by extension if specified
                if extensions is None or file_path.suffix.lower() in extensions:
                    files.append(file_path)
        
        if not files:
            logging.warning("No files found in the specified folder")
            return {}
        
        # Process each file
        for file_path in files:
            logging.info(f"Processing: {file_path.name}")
            
            # Read file content
            content = self.read_file(file_path)
            
            if content:
                # Count tokens
                token_count = self.count_tokens_in_text(content)
                
                # Store results
                relative_path = file_path.relative_to(self.folder_path)
                file_token_counts[str(relative_path)] = token_count
                all_token_counts.append(token_count)
                
                logging.info(f"  Tokens: {token_count}")
        
        # Calculate statistics
        if all_token_counts:
            stats = self.calculate_statistics(all_token_counts)
            
            results = {
                'folder': str(self.folder_path),
                'total_files': len(all_token_counts),
                'statistics': stats,
                'file_details': file_token_counts
            }
            
            return results
        else:
            logging.warning("No valid files were processed")
            return {}
    
    def calculate_statistics(self, token_counts: List[int]) -> Dict:
        """Calculate statistical measures for token counts."""
        # Calculate mode
        counter = Counter(token_counts)
        mode_data = counter.most_common(1)
        mode_value = mode_data[0][0] if mode_data else None
        mode_count = mode_data[0][1] if mode_data else 0
        
        stats = {
            'min': min(token_counts),
            'max': max(token_counts),
            'mean': statistics.mean(token_counts),
            'median': statistics.median(token_counts),
            'mode': mode_value,
            'mode_frequency': mode_count,
            'std_dev': statistics.stdev(token_counts) if len(token_counts) > 1 else 0,
            'total_tokens': sum(token_counts)
        }
        
        return stats
    
    def print_results(self, results: Dict):
        """Pretty print the analysis results."""
        if not results:
            print("No results to display")
            return
        
        print("\n" + "="*60)
        print(f"TOKEN ANALYSIS RESULTS")
        print("="*60)
        print(f"Folder: {results['folder']}")
        print(f"Total files analyzed: {results['total_files']}")
        print("\n" + "-"*60)
        print("STATISTICS:")
        print("-"*60)
        
        stats = results['statistics']
        print(f"Minimum tokens:      {stats['min']:,}")
        print(f"Maximum tokens:      {stats['max']:,}")
        print(f"Mean tokens:         {stats['mean']:,.2f}")
        print(f"Median tokens:       {stats['median']:,.0f}")
        print(f"Mode tokens:         {stats['mode']:,} (appears {stats['mode_frequency']} times)")
        print(f"Standard deviation:  {stats['std_dev']:,.2f}")
        print(f"Total tokens:        {stats['total_tokens']:,}")
        
        print("\n" + "-"*60)
        print("TOP 10 LARGEST FILES:")
        print("-"*60)
        
        # Sort files by token count
        sorted_files = sorted(results['file_details'].items(), 
                            key=lambda x: x[1], reverse=True)
        
        for i, (filename, token_count) in enumerate(sorted_files[:10], 1):
            print(f"{i:2d}. {filename:<40} {token_count:>10,} tokens")
    
    def save_to_csv(self, results: Dict, output_file: str = "token_analysis.csv"):
        """Save detailed results to a CSV file."""
        if not results:
            logging.warning("No results to save")
            return
        
        # Create DataFrame from file details
        df_data = []
        folder_path = ""
        for filename, token_count in results['file_details'].items():
            df_data.append({
                'filename': folder_path + filename,
                'token_count': token_count
            })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('token_count', ascending=False)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        logging.info(f"Detailed results saved to {output_file}")
        
        # Also save statistics to a separate file
        stats_df = pd.DataFrame([results['statistics']])
        stats_file = output_file.replace('.csv', '_statistics.csv')
        stats_df.to_csv(stats_file, index=False)
        logging.info(f"Statistics saved to {stats_file}")


def main():
    """Main function to run the token analysis."""
    # Example usage
    folder_path = input("Enter the folder path to analyze: ").strip()
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return
    
    # Ask for file extensions
    ext_input = input("Enter file extensions to analyze (comma-separated, e.g., .txt,.md,.py) or press Enter for all files: ").strip()
    
    extensions = None
    if ext_input:
        extensions = [ext.strip() for ext in ext_input.split(',')]
        # Ensure extensions start with a dot
        extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
    
    # Create analyzer
    analyzer = TokenAnalyzer(folder_path)
    
    # Run analysis
    results = analyzer.analyze_folder(extensions=extensions)
    
    # Print results
    analyzer.print_results(results)
    
    # Ask if user wants to save to CSV
    save_csv = input("\nSave detailed results to CSV? (y/n): ").strip().lower()
    if save_csv == 'y':
        output_file = input("Enter output filename (default: token_analysis.csv): ").strip()
        if not output_file:
            output_file = "token_analysis_plastics.csv"
        analyzer.save_to_csv(results, output_file)


if __name__ == "__main__":
    main()
