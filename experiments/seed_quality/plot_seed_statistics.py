import matplotlib.pyplot as plt
import numpy as np
import os
import json
import glob

def load_data(directory):
    data = {}
    for filepath in glob.glob(os.path.join(directory, "seed_stats_*.json")):
        filename = os.path.basename(filepath)
        # Extract topic and category from filename: seed_stats_topic_category.json
        parts = filename.replace("seed_stats_", "").replace(".json", "").split("_")
        if len(parts) >= 2:
            known_categories = [
                "scientific_paper", "news_article", "film_work", "music_work",
                "legal_case", "financial_report", "blog_post", "sport_match"
            ]
            
            topic = parts[0]
            category = "_".join(parts[1:])
            
            for cat in known_categories:
                if category.endswith(cat) and len(parts) > 2:
                    topic = "_".join(parts[:-len(cat.split("_"))])
                    category = cat
                    break
            
            with open(filepath, 'r') as f:
                content = json.load(f)
                stats = content.get("statistics", {})
                unique = stats.get("unique_seeds", 0)
                total = stats.get("successful_generations", 1)
                trials = stats.get("avg_trials_per_generation", 1) # Default to 1
                
                # Failed Proposal Rate q = (T - 1) / T
                failed_proposal_rate = (trials - 1) / trials * 100 if trials >= 1 else 0
                uniqueness = (unique / total) * 100 if total > 0 else 0
                
                # Format for consistency: (Topic, Uniqueness, Failed Proposal, Trials, Category)
                data[(topic, category)] = {
                    'topic': topic.replace("_", " ").capitalize(),
                    'uniqueness': uniqueness,
                    'failed_proposal_rate': failed_proposal_rate,
                    'trials': trials,
                    'category': category
                }
    return data

def get_group_name(category):
    mapping = {
        "scientific_paper": "Paper",
        "news_article": "News",
        "film_work": "Works",
        "music_work": "Works",
        "works": "Works",
        "institution": "Organization",
        "company": "Organization",
        "legal_case": "Document",
        "financial_report": "Document",
        "blog_post": "Blog",
        "restaurant": "Place",
        "place": "Place",
        "event": "Event",
        "sport_match": "Event",
        "person": "Person"
    }
    return mapping.get(category, category.replace("_", " ").capitalize())

def main():
    study_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(study_dir, "results")
    
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} does not exist.")
        return

    data = load_data(results_dir)
    
    # NEW GROUPING ORDER
    # Group B: Unique-Source Seed
    unique_source_groups = ["Paper", "News", "Document", "Blog"]
    # Group A: Common-Source Seed
    common_source_groups = ["Person", "Place", "Event", "Organization", "Works"]
    
    group_order = unique_source_groups + common_source_groups
    
    # Colors: More harmonious palettes matching the Failed Proposal Rate line (#E74C3C)
    # Red-tinted Purples and Deeper Oranges
    purple_palette = ['#884EA0', '#7D3C98', '#6C3483', '#512E5F'] 
    orange_palette = ['#E67E22', '#D35400', '#BA4A00', '#A04000', '#873600']
    
    line_color = '#000000' # A slightly more vibrant red
    
    group_colors = {}
    for i, g in enumerate(unique_source_groups):
        group_colors[g] = purple_palette[i % len(purple_palette)]
    for i, g in enumerate(common_source_groups):
        group_colors[g] = orange_palette[i % len(orange_palette)]
    
    # 1. GROUP DATA BY CATEGORY (AVERAGED OVER TOPICS)
    grouped_stats = {g: {'uniqueness': [], 'failed_proposal_rate': []} for g in group_order}
    detailed_data = {g: [] for g in group_order}

    for (topic_id, cat_id), values in data.items():
        gname = get_group_name(cat_id)
        if gname in grouped_stats:
            grouped_stats[gname]['uniqueness'].append(values['uniqueness'])
            grouped_stats[gname]['failed_proposal_rate'].append(values['failed_proposal_rate'])
            detailed_data[gname].append(values)

    avg_data = []
    for gname in group_order:
        if not grouped_stats[gname]['uniqueness']: continue
        avg_uniqueness = np.mean(grouped_stats[gname]['uniqueness'])
        avg_failed_proposal_rate = np.mean(grouped_stats[gname]['failed_proposal_rate'])
        avg_data.append({
            'label': gname,
            'uniqueness': avg_uniqueness,
            'failed_proposal_rate': avg_failed_proposal_rate,
            'color': group_colors[gname],
            'is_unique_source': gname in unique_source_groups
        })

    # --- PLOT 1: COMPACT PER CATEGORY (BAR + LINE/DOT) ---
    plt.rcParams.update({'font.size': 13, 'font.family': 'sans-serif'}) # Slightly smaller font for compactness
    fig1, ax1 = plt.subplots(figsize=(6, 3.6)) # More compact size
    
    x = np.arange(len(avg_data))
    
    # Use just one number scale as both use the same scale on the left with no label
    bars = ax1.bar(x, [d['uniqueness'] for d in avg_data], color=[d['color'] for d in avg_data], 
                  edgecolor='black', alpha=0.8, zorder=2, label='Uniqueness (%)')
    
    ax1.plot(x, [d['failed_proposal_rate'] for d in avg_data], color="#ffffff", marker='D', 
                 linewidth=2.5, markersize=8, zorder=5, label='Failed Proposal Rate (%)')
    
    ax1.set_ylim(0, 115) # Leave space for labels
    ax1.set_xticks(x)
    ax1.set_xticklabels([d['label'] for d in avg_data], fontweight='bold', fontsize=12, rotation=20, ha='right')
    
    # Add vertical separation line
    sep_idx = len(unique_source_groups) - 0.5
    ax1.axvline(x=sep_idx, color='gray', linestyle='--', alpha=0.5, zorder=1)
    # Add labels for groups, shifted up higher for more space
    ax1.text(sep_idx/2, 122, 'Anchored-Evidence', ha='center', fontweight='bold', fontsize=13, color='#7D3C98')
    ax1.text(sep_idx + (len(common_source_groups)/2), 122, 'Open-Evidence', ha='center', fontweight='bold', fontsize=13, color='#D35400')

    # Add value labels (rounded to 0 decimals)
    for i, d in enumerate(avg_data):
        ax1.text(i, d['uniqueness'] + 1, f"{d['uniqueness']:.0f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax1.text(i, d['failed_proposal_rate'] + 10, f"{d['failed_proposal_rate']:.0f}%", 
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color="#ffffff")

    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', alpha=0.8, label='Uniqueness (%)'),
        Line2D([0], [0], color=line_color, marker='D', markersize=8, label='Failed Proposal Rate (%)', linewidth=2.5)
    ]
    ax1.legend(handles=legend_elements, loc='lower left', frameon=True, fontsize=10)
    
    ax1.grid(axis='y', linestyle=':', alpha=0.4)
    plt.tight_layout()
    
    plot1_path = os.path.join(study_dir, 'seed_stats_category_avg.png')
    plot1_pdf_path = os.path.join(study_dir, 'seed_stats_category_avg.pdf')
    plt.savefig(plot1_path, dpi=300)
    plt.savefig(plot1_pdf_path, bbox_inches='tight')
    print(f"Saved Plot 1 to {plot1_path}")

    # --- PLOT 2: DETAILED BREAKDOWN (COMPACT) ---
    plot_items = []
    current_y = 0
    group_offsets = []
    
    for gname in group_order:
        items = sorted(detailed_data[gname], key=lambda x: x['topic'])
        if not items: continue
        
        # Small extra gap between super groups
        if gname == common_source_groups[0]:
            current_y += 0.5
            
        start_y = current_y
        # Category title y position
        title_y = current_y
        current_y += 0.5 # Space for the title
        
        for item in items:
            plot_items.append({
                'label': item['topic'],
                'uniqueness': item['uniqueness'],
                'failed_proposal_rate': item['failed_proposal_rate'],
                'group': gname,
                'y': current_y,
                'color': group_colors[gname]
            })
            current_y += 0.7 # More compact row spacing
        
        group_offsets.append({
            'name': gname,
            'start': start_y,
            'end': current_y - 0.7,
            'color': group_colors[gname],
            'title_y': title_y
        })
        current_y += 0.3 # Small space between groups

    # Adjust figure size: smaller width (6.0) and very tight height
    fig2, ax2 = plt.subplots(figsize=(6.0, current_y * 0.32 + 1.2))
    height = 0.55 # Slightly thicker bars for internal text
    
    # Plot category titles (Just the name, no "Category:")
    for ginfo in group_offsets:
        ax2.text(0, ginfo['title_y'], ginfo['name'], ha='left', va='center', 
                 fontweight='bold', fontsize=12, color=ginfo['color'], alpha=0.8)

    for item in plot_items:
        color = item['color']
        # Main bar for uniqueness
        ax2.barh(item['y'], item['uniqueness'], height=height, color=color, edgecolor='black', alpha=0.9, zorder=3)
        
        ax2.text(-0.5, item['y'], item['label'], ha='right', va='center', fontsize=12)
        
        # Uniqueness value text - INSIDE BAR, WHITE, NO %
        # Only show if bar is wide enough, otherwise put it outside
        if item['uniqueness'] > 8:
            ax2.text(item['uniqueness'] - 1.5, item['y'], f"{item['uniqueness']:.0f}", 
                     ha='right', va='center', fontsize=12, fontweight='bold', color='white', zorder=4)
        else:
            ax2.text(item['uniqueness'] + 1, item['y'], f"{item['uniqueness']:.0f}", 
                     va='center', fontsize=12, fontweight='bold', color='black', zorder=4)
        
        # Failed proposal rate marker (illustrate position)
        marker_style = 'D' # Diamond
        marker_size = 40 # Larger marker
        # Use white marker if inside bar, red if outside
        if item['failed_proposal_rate'] <= item['uniqueness']:
            ax2.scatter(item['failed_proposal_rate'], item['y'], marker=marker_style, s=marker_size, 
                        color='white', edgecolor='black', linewidth=0.7, zorder=5)
        else:
            ax2.scatter(item['failed_proposal_rate'], item['y'], marker=marker_style, s=marker_size, 
                        color='#C0392B', edgecolor='black', linewidth=0.7, zorder=5)

    ax2.set_xlim(0, 105) # Compact range
    ax2.invert_yaxis()
    ax2.set_xlabel("Percentage (%)", fontweight='bold', fontsize=12)
    ax2.set_yticks([])
    ax2.grid(axis='x', linestyle=':', alpha=0.4, zorder=0)
    
    # Two separate legends
    from matplotlib.lines import Line2D
    
    # Legend 1 (Top): Entity Sources (Color coding)
    source_elements = [
        Patch(facecolor='#7D3C98', edgecolor='black', alpha=0.9, label='Anchored-Evidence'),
        Patch(facecolor='#D35400', edgecolor='black', alpha=0.9, label='Open-Evidence')
    ]
    leg1 = ax2.legend(handles=source_elements, loc='lower center', bbox_to_anchor=(0.5, 1.02), 
               ncol=2, fontsize=11, frameon=False)
    plt.gca().add_artist(leg1)

    # Legend 2 (Bottom): Metrics (Uniqueness & Failure Rate)
    metric_elements = [
        Patch(facecolor='gray', edgecolor='black', alpha=0.8, label='Uniqueness'),
        Line2D([0], [0], marker='D', color='w', label='Failed Proposal Rate', 
               markerfacecolor='white', markeredgecolor='black', markersize=8, linestyle='None')
    ]
    ax2.legend(handles=metric_elements, loc='upper center', bbox_to_anchor=(0.5, -0.06), 
                       ncol=2, fontsize=11, frameon=False)

    for spine in ['top', 'right', 'left']:
        ax2.spines[spine].set_visible(False)

    plt.tight_layout(pad=0.1)
    plot2_path = os.path.join(study_dir, 'seed_stats_detailed_breakdown_v4.png')
    plot2_pdf_path = os.path.join(study_dir, 'seed_stats_detailed_breakdown_v4.pdf')
    plt.savefig(plot2_path, dpi=300)
    plt.savefig(plot2_pdf_path, bbox_inches='tight')
    print(f"Saved Plot 2 to {plot2_path}")

if __name__ == "__main__":
    main()
