import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle, FancyBboxPatch
import pandas as pd

class HierarchicalAgentVisualizer:
    def __init__(self):
        """Initialize the visualizer with test data."""
        # Routing data for Layer 1 agents (out of 1000 total routes)
        self.layer1_routing = [
            {'agent': 'L1-1', 'routes': 98, 'percentage': 9.8},
            {'agent': 'L1-2', 'routes': 95, 'percentage': 9.5},
            {'agent': 'L1-3', 'routes': 102, 'percentage': 10.2},
            {'agent': 'L1-4', 'routes': 89, 'percentage': 8.9},
            {'agent': 'L1-5', 'routes': 93, 'percentage': 9.3},
            {'agent': 'L1-6', 'routes': 97, 'percentage': 9.7},
            {'agent': 'L1-7', 'routes': 78, 'percentage': 7.8},  # Low routing
            {'agent': 'L1-8', 'routes': 104, 'percentage': 10.4},
            {'agent': 'L1-9', 'routes': 91, 'percentage': 9.1},
            {'agent': 'L1-10', 'routes': 95, 'percentage': 9.5}
        ]
        
        # Layer 2 routing data (2 agents per Layer 1 agent)
        self.layer2_routing = [
            {'agent': 'L2-1', 'routes': 49, 'percentage': 4.9, 'parent': 'L1-1'},
            {'agent': 'L2-2', 'routes': 49, 'percentage': 4.9, 'parent': 'L1-1'},
            {'agent': 'L2-3', 'routes': 47, 'percentage': 4.7, 'parent': 'L1-2'},
            {'agent': 'L2-4', 'routes': 48, 'percentage': 4.8, 'parent': 'L1-2'},
            {'agent': 'L2-5', 'routes': 51, 'percentage': 5.1, 'parent': 'L1-3'},
            {'agent': 'L2-6', 'routes': 51, 'percentage': 5.1, 'parent': 'L1-3'},
            {'agent': 'L2-7', 'routes': 44, 'percentage': 4.4, 'parent': 'L1-4'},
            {'agent': 'L2-8', 'routes': 45, 'percentage': 4.5, 'parent': 'L1-4'},
            {'agent': 'L2-9', 'routes': 46, 'percentage': 4.6, 'parent': 'L1-5'},
            {'agent': 'L2-10', 'routes': 47, 'percentage': 4.7, 'parent': 'L1-5'},
            {'agent': 'L2-11', 'routes': 48, 'percentage': 4.8, 'parent': 'L1-6'},
            {'agent': 'L2-12', 'routes': 49, 'percentage': 4.9, 'parent': 'L1-6'},
            {'agent': 'L2-13', 'routes': 39, 'percentage': 3.9, 'parent': 'L1-7'},  # Low
            {'agent': 'L2-14', 'routes': 39, 'percentage': 3.9, 'parent': 'L1-7'},  # Low
            {'agent': 'L2-15', 'routes': 52, 'percentage': 5.2, 'parent': 'L1-8'},
            {'agent': 'L2-16', 'routes': 52, 'percentage': 5.2, 'parent': 'L1-8'},
            {'agent': 'L2-17', 'routes': 45, 'percentage': 4.5, 'parent': 'L1-9'},
            {'agent': 'L2-18', 'routes': 46, 'percentage': 4.6, 'parent': 'L1-9'},
            {'agent': 'L2-19', 'routes': 47, 'percentage': 4.7, 'parent': 'L1-10'},
            {'agent': 'L2-20', 'routes': 48, 'percentage': 4.8, 'parent': 'L1-10'}
        ]
        
        # Test results summary
        self.test_summary = {
            'total_tests': 1000,
            'successful_routes': 942,
            'failed_routes': 58,
            'success_rate': 94.2,
            'avg_latency': 847,
            'avg_hops': 2.3,
            'layer1_utilization': 87,
            'layer2_utilization': 73,
            'error_rate': 2.1
        }
    
    def get_routing_color(self, percentage, layer=1):
        """Get color based on routing percentage."""
        if layer == 1:
            if percentage < 8.5:
                return '#e74c3c'  # Red - Low routing
            elif percentage < 9.5:
                return '#f39c12'  # Orange - Medium routing
            else:
                return '#2ecc71'  # Green - High routing
        else:  # Layer 2
            if percentage < 4.2:
                return '#e74c3c'  # Red - Low routing
            elif percentage < 4.8:
                return '#f39c12'  # Orange - Medium routing
            else:
                return '#2ecc71'  # Green - High routing
    
    def create_hierarchy_diagram(self):
        """Create the hierarchical structure diagram."""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Title
        ax.text(8, 9.5, 'Hierarchical Multi-Agent System Architecture', 
                fontsize=20, fontweight='bold', ha='center')
        
        # Principal Agent
        principal_circle = Circle((8, 8), 0.4, color='#c0392b', alpha=0.8)
        ax.add_patch(principal_circle)
        ax.text(8, 8, 'P', fontsize=14, fontweight='bold', 
                ha='center', va='center', color='white')
        ax.text(8, 7.3, '100%', fontsize=10, fontweight='bold', 
                ha='center', va='center', bbox=dict(boxstyle="round,pad=0.1", 
                facecolor='white', alpha=0.8))
        ax.text(8, 8.7, 'Principal Layer', fontsize=12, fontweight='bold', ha='center')
        
        # Layer 1 Agents
        ax.text(8, 6.2, 'Layer 1 (10 Agents)', fontsize=12, fontweight='bold', ha='center')
        layer1_x_positions = np.linspace(2, 14, 10)
        
        for i, (x_pos, data) in enumerate(zip(layer1_x_positions, self.layer1_routing)):
            color = self.get_routing_color(data['percentage'], layer=1)
            circle = Circle((x_pos, 5.5), 0.3, color=color, alpha=0.8)
            ax.add_patch(circle)
            
            # Agent label
            ax.text(x_pos, 5.5, f"L1-{i+1}", fontsize=8, fontweight='bold',
                    ha='center', va='center', color='white')
            
            # Percentage label
            ax.text(x_pos, 5.0, f"{data['percentage']}%", fontsize=8, fontweight='bold',
                    ha='center', va='center', bbox=dict(boxstyle="round,pad=0.1", 
                    facecolor='white', alpha=0.8))
            
            # Connection to principal
            ax.plot([8, x_pos], [7.6, 5.8], 'k-', alpha=0.3, linewidth=1)
        
        # Layer 2 Agents
        ax.text(8, 3.7, 'Layer 2 (20 Agents)', fontsize=12, fontweight='bold', ha='center')
        layer2_x_positions = np.linspace(1.5, 14.5, 20)
        
        for i, (x_pos, data) in enumerate(zip(layer2_x_positions, self.layer2_routing)):
            color = self.get_routing_color(data['percentage'], layer=2)
            circle = Circle((x_pos, 3), 0.2, color=color, alpha=0.8)
            ax.add_patch(circle)
            
            # Agent label
            ax.text(x_pos, 3, f"L2-{i+1}", fontsize=6, fontweight='bold',
                    ha='center', va='center', color='white')
            
            # Percentage label
            ax.text(x_pos, 2.5, f"{data['percentage']}%", fontsize=6, fontweight='bold',
                    ha='center', va='center', bbox=dict(boxstyle="round,pad=0.1", 
                    facecolor='white', alpha=0.8))
            
            # Connection to parent Layer 1 agent
            parent_index = i // 2
            parent_x = layer1_x_positions[parent_index]
            ax.plot([parent_x, x_pos], [5.2, 3.2], 'k-', alpha=0.3, linewidth=1)
        
        # Legend
        legend_y = 1.5
        ax.text(8, legend_y + 0.5, 'Performance Legend', fontsize=12, fontweight='bold', ha='center')
        
        # Low routing
        low_circle = Circle((4, legend_y), 0.15, color='#e74c3c', alpha=0.8)
        ax.add_patch(low_circle)
        ax.text(4.5, legend_y, 'Low Routing (<8.5% L1, <4.2% L2)', fontsize=10, va='center')
        
        # Medium routing
        med_circle = Circle((8, legend_y), 0.15, color='#f39c12', alpha=0.8)
        ax.add_patch(med_circle)
        ax.text(8.5, legend_y, 'Medium Routing (8.5-9.5% L1, 4.2-4.8% L2)', fontsize=10, va='center')
        
        # High routing
        high_circle = Circle((12, legend_y), 0.15, color='#2ecc71', alpha=0.8)
        ax.add_patch(high_circle)
        ax.text(12.5, legend_y, 'High Routing (>9.5% L1, >4.8% L2)', fontsize=10, va='center')
        
        plt.tight_layout()
        return fig
    
    def create_routing_distribution_chart(self):
        """Create bar chart showing routing distribution."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        agents = [data['agent'] for data in self.layer1_routing]
        routes = [data['routes'] for data in self.layer1_routing]
        percentages = [data['percentage'] for data in self.layer1_routing]
        
        colors = [self.get_routing_color(p, layer=1) for p in percentages]
        
        bars = ax.bar(agents, routes, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add percentage labels on top of bars
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{percentage}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Routing Distribution Across Layer 1 Agents', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Layer 1 Agents', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Routes', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(routes) + 10)
        
        # Add average line
        avg_routes = np.mean(routes)
        ax.axhline(y=avg_routes, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(len(agents)-1, avg_routes + 2, f'Average: {avg_routes:.1f}', 
                ha='right', va='bottom', fontweight='bold', color='red')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def create_metrics_dashboard(self):
        """Create metrics dashboard."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('System Performance Metrics Dashboard', fontsize=18, fontweight='bold')
        
        # Metric 1: Success Rate
        ax1.pie([self.test_summary['successful_routes'], self.test_summary['failed_routes']], 
                labels=['Successful', 'Failed'], autopct='%1.1f%%', 
                colors=['#2ecc71', '#e74c3c'], startangle=90)
        ax1.set_title(f"Success Rate: {self.test_summary['success_rate']}%", fontweight='bold')
        
        # Metric 2: Layer Utilization
        layers = ['Layer 1', 'Layer 2']
        utilization = [self.test_summary['layer1_utilization'], self.test_summary['layer2_utilization']]
        bars = ax2.bar(layers, utilization, color=['#3498db', '#2ecc71'], alpha=0.8)
        ax2.set_title('Layer Utilization (%)', fontweight='bold')
        ax2.set_ylabel('Utilization %')
        ax2.set_ylim(0, 100)
        
        for bar, util in zip(bars, utilization):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                     f'{util}%', ha='center', va='bottom', fontweight='bold')
        
        # Metric 3: Response Time Distribution (simulated)
        np.random.seed(42)
        response_times = np.random.normal(self.test_summary['avg_latency'], 150, 1000)
        ax3.hist(response_times, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax3.axvline(self.test_summary['avg_latency'], color='red', linestyle='--', linewidth=2)
        ax3.set_title(f"Response Time Distribution (Avg: {self.test_summary['avg_latency']}ms)", fontweight='bold')
        ax3.set_xlabel('Response Time (ms)')
        ax3.set_ylabel('Frequency')
        
        # Metric 4: Key Performance Indicators
        ax4.axis('off')
        metrics_text = f"""
        📊 KEY PERFORMANCE INDICATORS
        
        Total Test Cases: {self.test_summary['total_tests']:,}
        Success Rate: {self.test_summary['success_rate']}%
        Average Latency: {self.test_summary['avg_latency']}ms
        Average Routing Hops: {self.test_summary['avg_hops']}
        Error Rate: {self.test_summary['error_rate']}%
        
        🚨 ALERTS
        • L1-7 underperforming (7.8% vs 10% expected)
        • Error rate slightly above target (2.1% vs 2.0%)
        
        ✅ RECOMMENDATIONS
        • Investigate L1-7 capacity issues
        • Consider load balancing adjustment
        • Monitor L2-13 and L2-14 performance
        """
        ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def create_heatmap_analysis(self):
        """Create heatmap analysis of routing patterns."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Layer 1 routing heatmap
        layer1_data = np.array([data['percentage'] for data in self.layer1_routing]).reshape(1, -1)
        im1 = ax1.imshow(layer1_data, cmap='RdYlGn', aspect='auto', vmin=7, vmax=11)
        ax1.set_title('Layer 1 Routing Distribution Heatmap', fontweight='bold', pad=20)
        ax1.set_xticks(range(10))
        ax1.set_xticklabels([f'L1-{i+1}' for i in range(10)])
        ax1.set_yticks([])
        
        # Add percentage annotations
        for i in range(10):
            ax1.text(i, 0, f'{self.layer1_routing[i]["percentage"]}%', 
                     ha='center', va='center', fontweight='bold')
        
        # Layer 2 routing heatmap
        layer2_data = np.array([data['percentage'] for data in self.layer2_routing]).reshape(2, -1)
        im2 = ax2.imshow(layer2_data, cmap='RdYlGn', aspect='auto', vmin=3.5, vmax=5.5)
        ax2.set_title('Layer 2 Routing Distribution Heatmap', fontweight='bold', pad=20)
        ax2.set_xticks(range(10))
        ax2.set_xticklabels([f'L1-{i+1}\nChildren' for i in range(10)])
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Child 1', 'Child 2'])
        
        # Add percentage annotations for Layer 2
        for i in range(10):
            for j in range(2):
                idx = i * 2 + j
                ax2.text(i, j, f'{self.layer2_routing[idx]["percentage"]}%', 
                         ha='center', va='center', fontweight='bold', fontsize=9)
        
        # Add colorbars
        cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.1)
        cbar1.set_label('Routing Percentage (%)', fontweight='bold')
        
        cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.1)
        cbar2.set_label('Routing Percentage (%)', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def generate_comprehensive_report(self, save_plots=True, output_dir='agent_system_analysis'):
        """Generate comprehensive report with all visualizations."""
        import os
        
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
        
        print("🤖 Hierarchical Multi-Agent System Analysis Report")
        print("=" * 60)
        
        # Generate all visualizations
        print("📊 Generating hierarchy diagram...")
        fig1 = self.create_hierarchy_diagram()
        if save_plots:
            fig1.savefig(f'{output_dir}/hierarchy_diagram.png', dpi=300, bbox_inches='tight')
        
        print("📈 Generating routing distribution chart...")
        fig2 = self.create_routing_distribution_chart()
        if save_plots:
            fig2.savefig(f'{output_dir}/routing_distribution.png', dpi=300, bbox_inches='tight')
        
        print("📋 Generating metrics dashboard...")
        fig3 = self.create_metrics_dashboard()
        if save_plots:
            fig3.savefig(f'{output_dir}/metrics_dashboard.png', dpi=300, bbox_inches='tight')
        
        print("🔥 Generating heatmap analysis...")
        fig4 = self.create_heatmap_analysis()
        if save_plots:
            fig4.savefig(f'{output_dir}/heatmap_analysis.png', dpi=300, bbox_inches='tight')
        
        # Generate summary report
        self.print_summary_report()
        
        if save_plots:
            print(f"\n💾 All visualizations saved to '{output_dir}/' directory")
        
        plt.show()
        
        return {
            'hierarchy': fig1,
            'routing': fig2, 
            'metrics': fig3,
            'heatmap': fig4
        }
    
    def print_summary_report(self):
        """Print text-based summary report."""
        print("\n📋 EXECUTIVE SUMMARY")
        print("-" * 40)
        print(f"Total Test Cases: {self.test_summary['total_tests']:,}")
        print(f"Success Rate: {self.test_summary['success_rate']}%")
        print(f"Average Response Time: {self.test_summary['avg_latency']}ms")
        print(f"Average Routing Hops: {self.test_summary['avg_hops']}")
        
        print("\n🔍 PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        # Identify top and bottom performers
        sorted_l1 = sorted(self.layer1_routing, key=lambda x: x['percentage'])
        print(f"🟢 Best Performer: {sorted_l1[-1]['agent']} ({sorted_l1[-1]['percentage']}%)")
        print(f"🔴 Worst Performer: {sorted_l1[0]['agent']} ({sorted_l1[0]['percentage']}%)")
        
        print("\n⚠️  ALERTS & RECOMMENDATIONS")
        print("-" * 40)
        
        # Find underperforming agents
        avg_percentage = np.mean([data['percentage'] for data in self.layer1_routing])
        underperforming = [data for data in self.layer1_routing if data['percentage'] < avg_percentage - 1]
        
        if underperforming:
            print("🚨 Underperforming Agents:")
            for agent in underperforming:
                print(f"   • {agent['agent']}: {agent['percentage']}% (Expected: ~{avg_percentage:.1f}%)")
        
        print("\n💡 Recommendations:")
        print("   • Investigate L1-7 capacity and connection issues")
        print("   • Consider redistributing load from high-traffic agents")
        print("   • Monitor error rates on underperforming nodes")
        print("   • Implement proactive health checks")

# Example usage and demo
if __name__ == "__main__":
    # Create visualizer instance
    visualizer = HierarchicalAgentVisualizer()
    
    # Generate comprehensive report
    print("🚀 Starting Hierarchical Multi-Agent System Analysis...")
    figures = visualizer.generate_comprehensive_report(save_plots=True)
    
    # Optional: Create individual plots
    print("\n🎯 Individual plot generation examples:")
    print("• visualizer.create_hierarchy_diagram()")
    print("• visualizer.create_routing_distribution_chart()")
    print("• visualizer.create_metrics_dashboard()")
    print("• visualizer.create_heatmap_analysis()")