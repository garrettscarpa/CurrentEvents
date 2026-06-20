# CurrentEvents

*(previously `pClamp11_vClamp`)*

CurrentEvents is a GUI application for analyzing voltage clamp recordings collected in gapfree mode using pClamp11. By default it analyzes the first 60 seconds of each recording, but this duration can be adjusted in the app's detection settings.

## Installation

CurrentEvents is now packaged as a standalone application with PyInstaller, so **Python and the scientific dependencies (numpy, scipy, matplotlib, pyabf, PyQt5) no longer need to be installed separately.**

1. Download the latest build from the [Releases](../../releases) page of the `CurrentEvents` repository.
2. Unzip the download.
3. Launch the `CurrentEvents` executable.
   - **macOS:** if you get a security warning, right-click the app and choose *Open*, then confirm.
   - **Windows:** if SmartScreen appears, click *More info → Run anyway*.

Detection metrics and analysis options (including the 60 s window) are now configured inside the app rather than by editing a script.

## Organizing your data

Put the gapfree recordings in separate folders within one parent folder, where each subfolder title is the condition of its contents.

Once the data are correctly organized, point CurrentEvents at your parent directory (the one that holds the condition subfolders and data) and start the analysis after reviewing the detection metrics.

## Event detection and curation

Use the left and right arrows to cycle through all of the detected peaks. Peaks are accepted by default, but you can reject them with the down arrow on the keyboard. You can re-accept a rejected peak with the up arrow. You can drag the bases to adjust them if they don't properly align with your event.

<img width="1892" height="957" alt="Screenshot 2026-06-02 at 11 53 30" src="https://github.com/user-attachments/assets/dbb44d68-7170-426d-b3f8-5cfd80ea9edf" />

If you see a peak in the bottom window that wasn't detected, simply click the **Add Peak** button and then click the peak in the bottom window to add it automatically.

If you need to jump to a specific peak, enter that value in the **Go To #** window and press enter. **MAKE SURE YOU CLICK Q WHEN DONE TO SAVE THE DATA!!!!!!**

## Results

Once detection and adjustments are complete, run the results view to see automatically generated comparison graphs. If you want to separate into populations based on half-width (to separate duplicate events or gap-junction-like events), change `switch_by_halfwidth` from `True` to `False` and set the `halfwidth_threshold_ms` value.

The app generates 2 figures. The first shows the comparison features between the conditions, along with overlays of all events for each condition.

<img width="1845" height="950" alt="Screenshot 2026-06-02 at 12 04 06" src="https://github.com/user-attachments/assets/ba801603-38e0-470e-ac33-5862c1e88196" />

The second is a validation figure, which shows the sum of the onset and offset times for each condition (time to peak and time to recover). It compares this value to the difference in base values (on the x-axis), and plots the residual (which should be 0).

<img width="1034" height="538" alt="Screenshot 2026-06-02 at 12 04 53" src="https://github.com/user-attachments/assets/dd1af2a4-78b7-4224-bf7f-f477c36647cf" />

Note that if `split_by_halfwidth = True`, then 2x the graphs will be generated (one set of graphs for each group of events).
