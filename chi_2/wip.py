import pandas as pd
import matplotlib.pyplot as plt



# read in the example data
admissions = pd.read_csv("hospital_readmissions_3000.csv")


print(admissions.columns)


def plot_proportional(data,
                      bar_col,
                      target_col,
                      title,
                      xlabel, 
                      ylabel='Readmitted within 30 Days?',
                      filename):

    counts = pd.crosstab(data[bar_col], data[target_col])

    # 3. Convert counts to proportions (100% stacked) by dividing each row by its sum
    proportions = counts.div(counts.sum(axis=1), axis=0)

    # 4. Plot the stacked bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    proportions.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        linewidth=0.5,
    )

    ax.set_ylabel(ylabel)
    ax.set_xlabel('Post Discharge Care Required?')
    ax.set_title('Proportion of Patients Re-admitted within 30 days, by Post Discharge Care')
    ax.legend(title='Legend Categories', bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y))
    )

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        f"filename}.jpg', format='jpeg', dpi=300, bbox_inches='tight'
    )
    plt.show()


    'Post Discharge Care Required?'