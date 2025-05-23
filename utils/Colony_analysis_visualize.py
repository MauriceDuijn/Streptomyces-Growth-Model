import matplotlib.pyplot as plt


class CAVisualize:
    @staticmethod
    def show_PCA(projected, colony_index):
        plt.figure(figsize=(6, 6))
        plt.scatter(projected[colony_index][:, 0], projected[colony_index][:, 1], c="black")
        plt.axhline(0, color='gray', linewidth=0.8)
        plt.axvline(0, color='gray', linewidth=0.8)
        plt.xlabel("PC1 (major axis)")
        plt.ylabel("PC2 (minor axis)")
        plt.title(f"Colony [{colony_index}] Structure in PCA Frame")
        plt.axis("equal")
        plt.show()

    @staticmethod
    def show_propensity(distribution, colony_index):
        plt.figure()
        plt.hist(distribution, bins=20)
        plt.xlabel("Total Propensity per Cell", fontsize=12)
        plt.ylabel("Number of Cells", fontsize=12)
        plt.title(f"Colony {colony_index} Distribution of Cell Event Propensities", fontsize=14)
        plt.show()

    @staticmethod
    def show_crowding(distribution, colony_index, extra_text):
        plt.figure()
        plt.hist(distribution[colony_index], bins=20)
        plt.xlabel("Total Crowding per Cell", fontsize=12)
        plt.ylabel("Number of Cells", fontsize=12)
        plt.title(f"{extra_text}Colony {colony_index} Distribution of Cell Crowding", fontsize=14)
        plt.show()
