#!/usr/bin/env python3
"""
Advanced Visualization Enhancements for Academic Word Cloud Generation

This module provides enhanced visualization capabilities with scientific-grade
color schemes, layout algorithms, and temporal analysis features designed for
academic conferences.

Enhanced Features:
- Scientific color palettes optimized for publication
- Advanced layout algorithms for better readability
- Temporal analysis with trend visualization
- Export formats suitable for academic papers
- Accessibility-compliant color schemes
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any
import colorsys
from pathlib import Path


class AcademicColorSchemes:
    """Scientific-grade color schemes for academic publications"""
    
    @staticmethod
    def create_viridis_academic():
        """Viridis-inspired scheme optimized for word clouds"""
        colors = ['#440154', '#31688e', '#35b779', '#fde725']
        return LinearSegmentedColormap.from_list('viridis_academic', colors)
    
    @staticmethod 
    def create_nature_publication():
        """Color scheme following Nature publication guidelines"""
        colors = ['#1f4e79', '#2e86c1', '#58d68d', '#f4d03f']
        return LinearSegmentedColormap.from_list('nature_pub', colors)
    
    @staticmethod
    def create_ieee_compliant():
        """IEEE publication compliant color scheme"""
        colors = ['#0d47a1', '#1976d2', '#42a5f5', '#90caf9']
        return LinearSegmentedColormap.from_list('ieee_compliant', colors)
    
    @staticmethod
    def create_accessibility_friendly():
        """Colorblind-friendly academic color scheme"""
        colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b']
        return LinearSegmentedColormap.from_list('accessibility', colors)
    
    @staticmethod
    def create_monochrome_professional():
        """Professional monochrome scheme for print publications"""
        colors = ['#000000', '#404040', '#808080', '#c0c0c0']
        return LinearSegmentedColormap.from_list('mono_prof', colors)


class AcademicLayoutAlgorithms:
    """Advanced layout algorithms for academic word clouds"""
    
    @staticmethod
    def spiral_academic(center_x: float, center_y: float, radius: float, 
                       angle_step: float = 0.1) -> Tuple[float, float]:
        """Academic-optimized spiral layout for better readability"""
        # Fibonacci spiral for more natural distribution
        angle = 0
        r = 0
        golden_ratio = (1 + 5**0.5) / 2
        
        while r < radius:
            x = center_x + r * np.cos(angle)
            y = center_y + r * np.sin(angle)
            yield x, y
            
            angle += angle_step
            r += angle_step / golden_ratio
    
    @staticmethod
    def grid_academic(width: int, height: int, 
                     word_count: int) -> List[Tuple[int, int]]:
        """Grid-based layout optimized for academic presentations"""
        positions = []
        rows = int(np.sqrt(word_count))
        cols = (word_count + rows - 1) // rows
        
        cell_width = width // cols
        cell_height = height // rows
        
        for i in range(word_count):
            row = i // cols
            col = i % cols
            x = col * cell_width + cell_width // 2
            y = row * cell_height + cell_height // 2
            positions.append((x, y))
        
        return positions


class TemporalWordCloudAnalyzer:
    """Temporal analysis for word cloud evolution over time"""
    
    def __init__(self):
        self.temporal_data = {}
    
    def analyze_temporal_trends(self, frequencies_by_year: Dict[str, Counter],
                               top_n: int = 50) -> Dict[str, Any]:
        """Analyze temporal trends in word frequencies"""
        years = sorted(frequencies_by_year.keys())
        
        # Track word evolution
        word_trends = {}
        all_words = set()
        
        for year_freq in frequencies_by_year.values():
            all_words.update(word for word, _ in year_freq.most_common(top_n))
        
        for word in all_words:
            trends = []
            for year in years:
                freq = frequencies_by_year[year].get(word, 0)
                trends.append(freq)
            word_trends[word] = {
                'frequencies': trends,
                'years': years,
                'total': sum(trends),
                'peak_year': years[np.argmax(trends)] if trends else None,
                'trend_direction': self._calculate_trend_direction(trends)
            }
        
        return {
            'word_trends': word_trends,
            'years': years,
            'emerging_words': self._identify_emerging_words(word_trends, years),
            'declining_words': self._identify_declining_words(word_trends, years),
            'stable_words': self._identify_stable_words(word_trends)
        }
    
    def _calculate_trend_direction(self, frequencies: List[int]) -> str:
        """Calculate overall trend direction"""
        if len(frequencies) < 2:
            return 'stable'
        
        # Simple linear regression to determine trend
        x = np.arange(len(frequencies))
        y = np.array(frequencies)
        
        if np.sum(y) == 0:
            return 'stable'
        
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _identify_emerging_words(self, word_trends: Dict, years: List[str]) -> List[str]:
        """Identify words that emerged in recent years"""
        emerging = []
        if len(years) < 3:
            return emerging
        
        recent_years = years[-3:]  # Last 3 years
        
        for word, data in word_trends.items():
            if data['trend_direction'] == 'increasing':
                # Check if word appeared more in recent years
                total_freq = sum(data['frequencies'])
                recent_freq = sum(data['frequencies'][-3:])
                
                if total_freq > 0 and recent_freq / total_freq > 0.6:
                    emerging.append(word)
        
        return emerging[:10]  # Top 10 emerging words
    
    def _identify_declining_words(self, word_trends: Dict, years: List[str]) -> List[str]:
        """Identify words that are declining in usage"""
        declining = []
        if len(years) < 3:
            return declining
        
        for word, data in word_trends.items():
            if data['trend_direction'] == 'decreasing':
                declining.append(word)
        
        return declining[:10]  # Top 10 declining words
    
    def _identify_stable_words(self, word_trends: Dict) -> List[str]:
        """Identify consistently used words"""
        stable = []
        
        for word, data in word_trends.items():
            if data['trend_direction'] == 'stable' and data['total'] > 10:
                stable.append(word)
        
        return stable[:10]  # Top 10 stable words
    
    def create_temporal_visualization(self, temporal_analysis: Dict[str, Any],
                                    output_path: str = "out/temporal_analysis.png"):
        """Create temporal visualization for academic papers"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Temporal Word Cloud Analysis', fontsize=16, fontweight='bold')
        
        years = temporal_analysis['years']
        word_trends = temporal_analysis['word_trends']
        
        # 1. Top emerging words trend
        ax1 = axes[0, 0]
        emerging_words = temporal_analysis['emerging_words'][:5]
        for word in emerging_words:
            if word in word_trends:
                trends = word_trends[word]['frequencies']
                ax1.plot(years, trends, marker='o', label=word, linewidth=2)
        
        ax1.set_title('Emerging Words Trends', fontweight='bold')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Top declining words trend
        ax2 = axes[0, 1]
        declining_words = temporal_analysis['declining_words'][:5]
        for word in declining_words:
            if word in word_trends:
                trends = word_trends[word]['frequencies']
                ax2.plot(years, trends, marker='s', label=word, linewidth=2)
        
        ax2.set_title('Declining Words Trends', fontweight='bold')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Trend distribution
        ax3 = axes[1, 0]
        trend_counts = {'increasing': 0, 'decreasing': 0, 'stable': 0}
        for data in word_trends.values():
            trend_counts[data['trend_direction']] += 1
        
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        wedges, texts, autotexts = ax3.pie(
            trend_counts.values(), 
            labels=trend_counts.keys(),
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )
        ax3.set_title('Word Trend Distribution', fontweight='bold')
        
        # 4. Top stable words (bar chart)
        ax4 = axes[1, 1]
        stable_words = temporal_analysis['stable_words'][:8]
        stable_freqs = [word_trends[word]['total'] for word in stable_words if word in word_trends]
        
        bars = ax4.bar(range(len(stable_words)), stable_freqs, color='#3498db')
        ax4.set_title('Most Stable Words (Total Frequency)', fontweight='bold')
        ax4.set_xlabel('Words')
        ax4.set_ylabel('Total Frequency')
        ax4.set_xticks(range(len(stable_words)))
        ax4.set_xticklabels(stable_words, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, freq in zip(bars, stable_freqs):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{freq}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📈 Temporal analysis visualization saved: {output_path}")


class EnhancedWordCloudGenerator:
    """Enhanced word cloud generator with academic-grade features"""
    
    def __init__(self):
        self.color_schemes = AcademicColorSchemes()
        self.layout_algorithms = AcademicLayoutAlgorithms()
        self.temporal_analyzer = TemporalWordCloudAnalyzer()
    
    def generate_academic_wordcloud(self, frequencies: Counter,
                                  output_path: str,
                                  scheme: str = 'nature_publication',
                                  title: str = "Academic Word Cloud",
                                  subtitle: str = None,
                                  max_words: int = 200,
                                  width: int = 1200,
                                  height: int = 800) -> None:
        """Generate academic-grade word cloud with enhanced features"""
        
        # Import here to avoid circular imports
        from wordcloud import WordCloud
        import matplotlib.font_manager as fm
        
        # Get color scheme
        colormap = getattr(self.color_schemes, f'create_{scheme}')()
        
        # Create custom color function
        def academic_color_func(word, font_size, position, orientation, 
                              random_state=None, **kwargs):
            # Map font size to color intensity
            normalized_size = (font_size - 10) / (100 - 10)  # Normalize to 0-1
            color_val = colormap(normalized_size)
            # Convert to RGB
            return tuple(int(c * 255) for c in color_val[:3])
        
        # Configure WordCloud with academic settings
        wordcloud = WordCloud(
            width=width,
            height=height,
            max_words=max_words,
            relative_scaling=0.6,  # Better visual hierarchy
            min_font_size=12,      # Readable minimum size
            max_font_size=100,     # Professional maximum size
            background_color='white',
            color_func=academic_color_func,
            random_state=42,       # Reproducibility
            collocations=False,    # Avoid unwanted combinations
            prefer_horizontal=0.8, # Academic readability
            margin=20,             # Professional margins
            mode='RGBA'            # Support transparency
        )
        
        # Generate word cloud
        wordcloud.generate_from_frequencies(frequencies)
        
        # Create figure with academic styling
        fig, ax = plt.subplots(1, 1, figsize=(width/100, height/100), dpi=300)
        
        # Display word cloud
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        # Add academic title
        if title:
            ax.set_title(title, fontsize=20, fontweight='bold', pad=30)
        
        if subtitle:
            ax.text(0.5, 0.95, subtitle, transform=ax.transAxes,
                   ha='center', va='top', fontsize=14, style='italic')
        
        # Add metadata footer
        metadata_text = f"Generated with {max_words} words | Scheme: {scheme}"
        ax.text(0.02, 0.02, metadata_text, transform=ax.transAxes,
               fontsize=10, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"🎨 Academic word cloud saved: {output_path}")
    
    def generate_temporal_series(self, frequencies_by_year: Dict[str, Counter],
                               output_dir: str = "out/temporal_wordclouds",
                               scheme: str = 'nature_publication') -> None:
        """Generate temporal series of word clouds for longitudinal analysis"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate individual year word clouds
        for year, frequencies in frequencies_by_year.items():
            if len(frequencies) > 0:
                output_file = output_path / f"wordcloud_{year}_{scheme}.png"
                self.generate_academic_wordcloud(
                    frequencies=frequencies,
                    output_path=str(output_file),
                    scheme=scheme,
                    title=f"Word Cloud Analysis - {year}",
                    subtitle=f"Top {min(200, len(frequencies))} words"
                )
        
        # Generate temporal analysis
        temporal_analysis = self.temporal_analyzer.analyze_temporal_trends(
            frequencies_by_year
        )
        
        # Create temporal visualization
        self.temporal_analyzer.create_temporal_visualization(
            temporal_analysis,
            output_path / "temporal_trend_analysis.png"
        )
        
        print(f"📊 Temporal series generated in: {output_dir}")
        return temporal_analysis


def demonstrate_academic_features():
    """Demonstrate academic word cloud features"""
    print("🎓 Demonstrating Academic Word Cloud Features")
    
    # Sample data for demonstration
    sample_frequencies = Counter({
        'computational': 100, 'linguistics': 95, 'analysis': 90,
        'algorithm': 85, 'research': 80, 'academic': 75,
        'publication': 70, 'conference': 65, 'paper': 60,
        'methodology': 55, 'evaluation': 50, 'framework': 45,
        'scientific': 40, 'reproducible': 35, 'benchmark': 30
    })
    
    generator = EnhancedWordCloudGenerator()
    
    # Test different academic color schemes
    schemes = ['nature_publication', 'ieee_compliant', 'viridis_academic', 
               'accessibility_friendly', 'monochrome_professional']
    
    output_dir = Path("out/academic_demos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for scheme in schemes:
        output_file = output_dir / f"demo_{scheme}.png"
        generator.generate_academic_wordcloud(
            frequencies=sample_frequencies,
            output_path=str(output_file),
            scheme=scheme,
            title=f"Academic Demo - {scheme.replace('_', ' ').title()}",
            subtitle="Demonstration of academic word cloud features"
        )
    
    print("✅ Academic features demonstration complete")


if __name__ == "__main__":
    demonstrate_academic_features()