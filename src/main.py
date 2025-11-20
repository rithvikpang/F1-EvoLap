from track.track_loader import Track
from track.track_visualizer import plot_track

def main():
    track = Track("data/silverstone.csv")

    print("Track width stats:")
    print("Min:", track.track_width.min())
    print("Max:", track.track_width.max())
    print("Mean:", track.track_width.mean())

    plot_track(track)

if __name__ == "__main__":
    main()