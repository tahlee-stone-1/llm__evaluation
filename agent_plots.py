import plotly.graph_objects as go
import numpy as np

class HierarchicalAgentPlotly:
    def __init__(self):
        """Initialize the Plotly visualizer with test data for 2 layers only."""
        # Agent performance data with F1, Precision, and Recall metrics
        self.layer1_performance = [
            {'agent': 'Order Management Agent', 'f1_score': 0.89, 'precision': 0.92, 'recall': 0.86, 'routes': 98},
            {'agent': 'Refund Processing Agent', 'f1_score': 0.87, 'precision': 0.90, 'recall': 0.84, 'routes': 95},
            {'agent': 'Product Information Agent', 'f1_score': 0.93, 'precision': 0.95, 'recall': 0.91, 'routes': 102},
            {'agent': 'Shipping Support Agent', 'f1_score': 0.82, 'precision': 0.85, 'recall': 0.79, 'routes': 89},
            {'agent': 'Payment Processing Agent', 'f1_score': 0.85, 'precision': 0.88, 'recall': 0.82, 'routes': 93},
            {'agent': 'Technical Support Agent', 'f1_score': 0.88, 'precision': 0.91, 'recall': 0.85, 'routes': 97},
            {'agent': 'Account Management Agent', 'f1_score': 0.74, 'precision': 0.78, 'recall': 0.71, 'routes': 78},
            {'agent': 'General Inquiry Agent', 'f1_score': 0.94, 'precision': 0.96, 'recall': 0.92, 'routes': 104},
            {'agent': 'Complaint Resolution Agent', 'f1_score': 0.83, 'precision': 0.86, 'recall': 0.80, 'routes': 91},
            {'agent': 'Policy Information Agent', 'f1_score': 0.87, 'precision': 0.90, 'recall': 0.84, 'routes': 95}
        ]
        
        # Professional Colors
        self.colors = {
            'principal': '#2C3E50',  # Navy Blue
            'low': '#e74c3c',        # Red - Low performance
            'medium': '#f39c12',     # Orange - Medium performance
            'high': '#2ecc71',       # Green - High performance
            'connections': '#95A5A6'  # Professional Grey
        }
    
    def get_f1_color(self, f1_score):
        """Get color based on F1 score."""
        if f1_score < 0.80:
            return self.colors['low']     # Low performance
        elif f1_score < 0.90:
            return self.colors['medium']  # Medium performance
        else:
            return self.colors['high']    # High performance
    
    def create_hierarchy_diagram(self, principal_font_size=16, agent_font_size=14):
        """Create the hierarchical structure diagram using Plotly."""
        fig = go.Figure()
        
        # Calculate overall F1 score for principal
        overall_f1 = np.mean([data['f1_score'] for data in self.layer1_performance])
        
        # Principal Agent
        principal_x, principal_y = 0, 1
        fig.add_trace(go.Scatter(
            x=[principal_x],
            y=[principal_y],
            mode='markers+text',
            marker=dict(
                size=80,  # Larger size
                color=self.colors['principal'],
                line=dict(color='white', width=3)
            ),
            text=[f'{overall_f1:.2f}'],  # Overall F1 score in the middle
            textfont=dict(color='white', size=principal_font_size, family='Calibri', 
                         style='normal', variant='normal'),
            textposition='middle center',
            hovertemplate=f'<b>Principal Agent</b><br>Overall F1: {overall_f1:.3f}<br>Role: Central Router & Coordinator<extra></extra>',
            name='Principal',
            showlegend=False
        ))
        
        # Principal summary statistics (offset to the right, no label)
        total_routes_sum = sum([data['routes'] for data in self.layer1_performance])
        avg_precision = np.mean([data['precision'] for data in self.layer1_performance])
        avg_recall = np.mean([data['recall'] for data in self.layer1_performance])
        principal_summary = f'<b>Routes:</b> 100% ({total_routes_sum})<br><b>Avg Precision:</b> {avg_precision:.1f}<br><b>Avg Recall:</b> {avg_recall:.1f}'
        
        fig.add_trace(go.Scatter(
            x=[principal_x + 0.3],  # Offset to the right
            y=[principal_y - 0.20],  # Position below principal node
            mode='text',
            text=[principal_summary],
            textfont=dict(color='black', size=10, family='Calibri'),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Layer 1 Agents - increased spacing for longer names
        layer1_x_positions = np.linspace(-4.5, 4.5, 10)  # Increased spacing from -3,3 to -4.5,4.5
        layer1_y = 0
        
        # Calculate total routes for percentage calculation
        total_routes = sum([data['routes'] for data in self.layer1_performance])
        
        for i, (x_pos, data) in enumerate(zip(layer1_x_positions, self.layer1_performance)):
            color = self.get_f1_color(data['f1_score'])
            route_percentage = (data['routes'] / total_routes) * 100
            
            # Agent circle with F1 score inside
            fig.add_trace(go.Scatter(
                x=[x_pos],
                y=[layer1_y],
                mode='markers+text',
                marker=dict(
                    size=60,  # Larger size
                    color=color,
                    line=dict(color='white', width=3)
                ),
                text=[f'{data["f1_score"]:.2f}'],  # F1 score in the middle
                textfont=dict(color='white', size=agent_font_size, family='Calibri', 
                             style='normal', variant='normal'),  # Changed to white and bold
                textposition='middle center',
                hovertemplate=f'<b>{data["agent"]}</b><br>' +
                            f'F1 Score: {data["f1_score"]:.3f}<br>' +
                            f'Precision: {data["precision"]:.3f}<br>' +
                            f'Recall: {data["recall"]:.3f}<br>' +
                            f'Routes: {data["routes"]}<extra></extra>',
                name=data['agent'],
                showlegend=False
            ))
            
            # Agent name below the circle in bold black (smaller font for longer names)
            fig.add_trace(go.Scatter(
                x=[x_pos],
                y=[layer1_y - 0.18],  # Moved down more
                mode='text',
                text=[f'<b>{data["agent"]}</b>'],  # Made bold
                textfont=dict(color='black', size=10, family='Calibri'),  # Reduced from 12 to 10
                hoverinfo='skip',
                showlegend=False
            ))
            
            # Summary statistics box below agent name (bold labels only, with route percentage)
            summary_text = f'<b>Routes:</b> {route_percentage:.1f}% ({data["routes"]})<br><b>Precision:</b> {data["precision"]:.1f}<br><b>Recall:</b> {data["recall"]:.1f}'
            fig.add_trace(go.Scatter(
                x=[x_pos],
                y=[layer1_y - 0.32],  # Position below agent name
                mode='text',
                text=[summary_text],
                textfont=dict(color='black', size=9, family='Calibri'),
                hoverinfo='skip',
                showlegend=False
            ))
        
        # Connection lines - Tree structure
        # Main trunk from Principal
        trunk_y_start = principal_y - 0.05
        trunk_y_end = 0.6
        fig.add_trace(go.Scatter(
            x=[principal_x, principal_x],
            y=[trunk_y_start, trunk_y_end],
            mode='lines',
            line=dict(color=self.colors['connections'], width=1.5),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Horizontal distribution line
        distribution_y = 0.4
        fig.add_trace(go.Scatter(
            x=[layer1_x_positions[0], layer1_x_positions[-1]],
            y=[distribution_y, distribution_y],
            mode='lines',
            line=dict(color=self.colors['connections'], width=1),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Vertical connector from trunk to distribution
        fig.add_trace(go.Scatter(
            x=[principal_x, principal_x],
            y=[trunk_y_end, distribution_y],
            mode='lines',
            line=dict(color=self.colors['connections'], width=1),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Individual branches to Layer 1 agents
        for x_pos in layer1_x_positions:
            fig.add_trace(go.Scatter(
                x=[x_pos, x_pos],
                y=[distribution_y, layer1_y + 0.05],
                mode='lines',
                line=dict(color=self.colors['connections'], width=0.8),
                hoverinfo='skip',
                showlegend=False
            ))
        
        # Add only Principal layer label
        fig.add_annotation(
            x=0, y=1.25,
            text="<b>Principal Layer</b>",  # Made bold
            showarrow=False,
            font=dict(size=16, color=self.colors['principal'], family='Calibri')
        )
        
        # Create custom legend
        legend_traces = [
            go.Scatter(x=[None], y=[None], mode='markers', 
                      marker=dict(size=15, color=self.colors['principal']),
                      name='Principal', showlegend=True),
            go.Scatter(x=[None], y=[None], mode='markers', 
                      marker=dict(size=15, color=self.colors['high']),
                      name='High Performance (F1 ≥ 0.90)', showlegend=True),
            go.Scatter(x=[None], y=[None], mode='markers', 
                      marker=dict(size=15, color=self.colors['medium']),
                      name='Medium Performance (0.80 ≤ F1 < 0.90)', showlegend=True),
            go.Scatter(x=[None], y=[None], mode='markers', 
                      marker=dict(size=15, color=self.colors['low']),
                      name='Low Performance (F1 < 0.80)', showlegend=True)
        ]
        
        for trace in legend_traces:
            fig.add_trace(trace)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='<b>Hierarchical Multi-Agent System Architecture</b>',  # Made bold
                font=dict(size=20, color=self.colors['principal'], family='Calibri'),
                x=0.5
            ),
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                range=[-5, 5]  # Increased range for wider spacing
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                range=[-0.5, 1.4]  # Increased bottom range for summary boxes
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=1.02,
                font=dict(size=10, color=self.colors['principal'], family='Calibri')
            ),
            margin=dict(l=50, r=150, t=80, b=80),  # Increased bottom margin
            width=1400,  # Increased width for longer agent names
            height=600
        )
        
        return fig

# Simple function to display the chart
def display_hierarchy(principal_font_size=16, agent_font_size=14):
    """Display hierarchy diagram with customizable font sizes."""
    visualizer = HierarchicalAgentPlotly()
    fig = visualizer.create_hierarchy_diagram(principal_font_size, agent_font_size)
    fig.show()
    return fig

# Example usage
if __name__ == "__main__":
    # Create and display the hierarchy diagram
    visualizer = HierarchicalAgentPlotly()
    fig = visualizer.create_hierarchy_diagram(principal_font_size=18, agent_font_size=16)
    fig.show()