import pandas as pd
import os
import matplotlib.pyplot as plt

def print_table(value_counts, name):
    table = value_counts.reset_index()
    table.columns = ['cmp_code', 'Frequency']

    fig, ax = plt.subplots(figsize=(6, len(table) * 0.1)) 
    ax.axis('tight')
    ax.axis('off')
    table_obj = ax.table(cellText=table.values, colLabels=table.columns, cellLoc='center', loc='center')
    plt.savefig(name, bbox_inches='tight', dpi=300)             


def show_csv_bar(value_counts):
    
    plt.figure(figsize=(10, 5))
    plt.bar(value_counts.index, value_counts.values, width=0.6, edgecolor='black')  # Width < 1 adds space
    plt.xticks(rotation='vertical')
    plt.xlabel('cmp_code')
    plt.ylabel('Frequency')
    plt.title('Histogram of cmp_code')
    plt.show()


def analyze_csv(file_path):
    df = pd.read_csv(file_path)
    data = df['cmp_code']
    data = data.str[:3]

    value_counts = data.value_counts()
    print("Count ordered by frequency")
    print(value_counts)
    #show_csv_bar(value_counts)
    print_table(value_counts, name="table_frequencies_raw_ordered_count.png")

    data = data.sort_values()
    value_counts = data.value_counts().sort_index()
    print("Count ordered by category name")
    print(value_counts)
    #show_csv_bar(value_counts) 
    print_table(value_counts, name="table_frequencies_raw_ordered_category_name.png")
     
        

if __name__ == "__main__":
    analyze_csv('CMPD_all_raw.csv')