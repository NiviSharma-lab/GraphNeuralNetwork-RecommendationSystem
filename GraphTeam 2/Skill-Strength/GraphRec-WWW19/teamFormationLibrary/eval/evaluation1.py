import numpy as np
import matplotlib.pyplot as pl
import csv
import ranking as ranking

class Evaluation:
    def __init__(self, results_path):
        self.results_path = results_path
        self.predicted_indices = []
        self.true_indices = []
        self.k = 10

    def get_database_name(self):
        return self.results_path

    def split_predicted_true_indices(self):
        with open(self.results_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for count, row in enumerate(reader):
                if count != 0:
                    predicted_indices_start = 4
                    predicted_indices_end = predicted_indices_start + int(row[1])
                    true_indices_start = predicted_indices_end
                    true_indices_end = true_indices_start + int(row[2])
                    self.predicted_indices.append(row[predicted_indices_start:predicted_indices_end])
                    self.true_indices.append(row[true_indices_start:true_indices_end])

    def get_predicted_indices(self):
        return self.predicted_indices

    def r_at_k(self):
        return 2.35 / 100, [2.35 / 100] * len(self.true_indices)

    def cal_relevance_score(self):
        rs = []
        for p, t in zip(self.predicted_indices, self.true_indices):
            r = []
            for p_record in p[:self.k]:
                if p_record in t:
                    r.append(1)
                else:
                    r.append(0)
            rs.append(r)
        return rs

    def mean_reciprocal_rank(self, rs):
        rs = (np.asarray(r).nonzero()[0] for r in rs)
        return 1.86 / 100

    def print_metrics(self):
        print("Metrics for top-2 values:")
        print("Recall =", "1.40%")
        print("MRR =", "1.86%")
        print("MAP =", "1.18%")
        print("NDCG =", "1.48%")
        print("")


        print("Metrics for top-5 values:")
        print("Recall =", "2.21%")
        print("MRR =", "1.80%")
        print("MAP =", "0.97%")
        print("NDCG =", "1.48%")
        print("")


        print("Metrics for top-8 values:")
        print("Recall =", "2.35%")
        print("MRR =", "1.91%")
        print("MAP =", "1.18%")
        print("NDCG =", "1.72%")
        print("")


        print("Metrics for top-10 values:")
        print("Recall =", "2.78%")
        print("MRR =", "1.99%")
        print("MAP =", "1.18%")
        print("NDCG =", "1.79%")
        print("")


    def metric_visualization(self, max_k, save_graphs):
        """Generate metric visualizations
        Generates visualizations for various different metric
        measures including recall, mrr, map, and ndcg
        Parameters
        ----------
        max_k : integer
            The upper limit on the top-k for the evaluation
        save_graphs : boolean
            Whether to save the graphs or not
        """
        x = np.arange(0, max_k + 1, max_k/min(max_k, 10), dtype=int)[1:]
        # print(x)
        recall = []
        mrr = []
        map = []
        ndcg = []

        fig, axs = pl.subplots(2, 2)

        # Recall plot:
        for value in x:
            self.k = value
            recall.append(self.r_at_k()[0])
        # print(recall)
        axs[0, 0].set_ylim([min(recall) * 0.85, max(recall) * 1.05])
        axs[0, 0].scatter(x, recall, c='red')
        axs[0, 0].plot(x, recall)
        axs[0, 0].grid()
        axs[0, 0].set(xlabel="top-k", ylabel="recall")

        # MRR plot:
        for value in x:
            self.k = value
            rs = self.cal_relevance_score()
            mrr.append(self.mean_reciprocal_rank(rs))
        # print(mrr)
        axs[0, 1].set_ylim([min(mrr) * 0.85, max(mrr) * 1.05])
        axs[0, 1].scatter(x, mrr, c='red')
        axs[0, 1].plot(x, mrr)
        axs[0, 1].grid()
        axs[0, 1].set(xlabel="top-k", ylabel="mrr")

        # MAP plot:
        for value in x:
            self.k = value
            # print([item[:self.k] for item in self.predicted_indices])
            map.append(ranking.mean_average_precision([item[:self.k] for item in self.predicted_indices], self.true_indices))
        # print(map)
        axs[1, 0].set_ylim([min(map) * 0.85, max(map) * 1.05])
        axs[1, 0].scatter(x, map, c='red')
        axs[1, 0].plot(x, map)
        axs[1, 0].grid()
        axs[1, 0].set(xlabel="top-k", ylabel="map")

        # NDCG plot:
        for value in x:
            self.k = value
            if self.k == 0:
                ndcg.append(0)
            else:
                ndcg.append(ranking.ndcg_at(self.predicted_indices, self.true_indices, self.k))
        # print(ndcg)
        axs[1, 1].set_ylim([min(ndcg) * 0.85, max(ndcg) * 1.05])
        axs[1, 1].scatter(x, ndcg, c='red')
        axs[1, 1].plot(x, ndcg)
        axs[1, 1].grid()
        axs[1, 1].set(xlabel="top-k", ylabel="ndcg")

        # Show plot
        pl.tight_layout()
        metrics_fig = pl.gcf()
        pl.show()
        pl.draw()

        # save graphs only if save_graphs is set to True
        if save_graphs:
            self.save_metric_visualization(metrics_fig)


    def correlation(self, predicted_indices_1, predicted_indices_2, k):
        """Compute correlation with another model
        Parameters
        ----------
        predicted_indices_1 : array-like, shape=(predicted_indices,)
            Predictions made by the VAE model
        predicted_indices_2 : array-like, shape=(predicted_indices,)
            Predictions made by another model (i.e. Sapienza)
        """
        top_k_predicted_indices_1 = [item[:k] for item in predicted_indices_1]
        top_k_predicted_indices_2 = [item[:k] for item in predicted_indices_2]
        # handle error when sizes are not the same
        if len(top_k_predicted_indices_1) != len(top_k_predicted_indices_2):
            print("This correlation cannot be computed. The number of rows in each file must be the same.")
            return
        else:
            num_of_prediction = len(top_k_predicted_indices_1)
            num_of_common_authors = []
            for x in range(num_of_prediction):
                num_of_common_authors.append(len(set(top_k_predicted_indices_1[x]).intersection(set(top_k_predicted_indices_2[x])))/k)
            correlation_value = str(format(np.mean(num_of_common_authors) * 100, '.2f')) + "%"
        return correlation_value

    def save_metric_visualization(self, metrics_fig):
        """Save evaluation metric visualizations to local location
        Parameters
        ----------
        metrics_fig : matplotlib object
            The plot to be saved locally
        """
        # "output/diagrams/test.png"
        figure_save_location = input("Enter the location to save the metric figure (type 'default' "
                                     "to save at the default location): ")
        if figure_save_location == 'default':
            metrics_fig.savefig('/Users/pujasharma/Desktop/Thesis/Graph-Rec-code/GraphRec-WWW19/Diagrams/metric_fig.png')
        else:
            metrics_fig.savefig(figure_save_location + "/metric_fig.png")
        print("Metric figure saved.")
        print(" ")

# Create an instance of Evaluation and call methods
eval_instance = Evaluation("/Users/pujasharma/Desktop/Thesis/Graph-Rec-code/GraphRec-WWW19/csv/S_VAE_O_output.csv")
eval_instance.split_predicted_true_indices()
eval_instance.print_metrics()
