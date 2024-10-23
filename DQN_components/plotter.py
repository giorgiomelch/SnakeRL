import matplotlib.pyplot as plt
import numpy as np

def plot_trend(scores, max_y=60, extra_title=None):
    calcola_media = lambda i: sum(scores[i-50:i]) / 50
    media_precedenti = np.array([calcola_media(i) for i in range(50, len(scores) + 1)])
    max_mean_value = np.max(media_precedenti)
    max_mean_index = np.argmax(media_precedenti) + 50  
    plt.plot(scores, label='Score')
    plt.plot(range(50, len(scores) + 1), media_precedenti, label='Mean score delle ultime 50 partite')
    plt.text(max_mean_index, max_mean_value, f'{max_mean_value:.2f}', fontsize=12, color="darkorange", ha='center')
    plt.title(f"Andamento del training {extra_title}")
    plt.xlabel("Partite")
    plt.ylabel("Score")
    plt.ylim(0, max_y)
    plt.legend()
    plt.grid()
    plt.show()