import pandas as pd
import matplotlib.pyplot as plt

if __name__=='__main__':
    seasons = {"annual": range(1,366), "summer":range(153,246)}

    target_cmaqs = pd.read_csv("D:/split-by-day/us-2011-satellite-target4-day-1.csv")[["cmaq_x", "cmaq_y", "cmaq_id"]].set_index("cmaq_id")

    # for model_name in ["LCF", "LCF_coord", "NNW", "NNW_SWA"]:
    #     save_dir = f"D:/prediction map/east-half/{model_name}/"
    #     for season_name in seasons.keys():
    #         all_preds = []
    #         reason_dates = seasons[season_name]
    #         for date in reason_dates:
    #             date_pred = pd.read_csv(f"{save_dir}date{date}_prediction.csv", index_col=0)
    #             date_pred.columns = [f"date{date}"]
    #             all_preds.append(date_pred)
    #         all_preds = pd.concat(all_preds, axis=1)
    #         mean_pred = all_preds.mean(axis=1)
    #         pred_xy = target_cmaqs.loc[all_preds.index]
    #         plt.scatter(pred_xy["cmaq_x"], pred_xy["cmaq_y"], c=mean_pred, s=4, cmap="rainbow", vmin=4, vmax=16, marker="s")
    #         plt.title(f"{model_name} {season_name} mean prediction")
    #         plt.axis('off')
    #         plt.colorbar()
    #         plt.savefig(f"{save_dir}{model_name}_{season_name}_mean prediction square")
    #         plt.close()

    for model_name in ["RF"]:
        save_dir = f"D:/prediction map/{model_name}/LCF/"
        for season_name in seasons.keys():
            all_preds = []
            reason_dates = seasons[season_name]
            for date in reason_dates:
                date_pred = pd.read_csv(f"{save_dir}date{date}_prediction.csv", index_col=0)
                date_pred.columns = [f"date{date}"]
                all_preds.append(date_pred)
            all_preds = pd.concat(all_preds, axis=1)
            mean_pred = all_preds.mean(axis=1)
            pred_xy = target_cmaqs.loc[all_preds.index]
            plt.scatter(pred_xy["cmaq_x"], pred_xy["cmaq_y"], c=mean_pred, s=4, cmap="rainbow", vmin=4, vmax=16)
            plt.title(f"{model_name} {season_name} mean prediction")
            plt.axis('off')
            plt.colorbar()
            plt.savefig(f"{save_dir}{model_name}_{season_name}_mean prediction")
            plt.close()
