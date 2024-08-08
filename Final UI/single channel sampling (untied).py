# single channel sampling (untied)
# analog output and input tasks untied from the main thread. The 2 ports act independently.
# Cant perform only 1 analog input port can be activated at a single time.
# The multichannel not working.

import sys
import numpy as np
import time
from functools import partial
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QGridLayout, 
    QScrollArea, QComboBox, QLineEdit, QFileDialog, QMessageBox, QProgressDialog, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRunnable, QThreadPool, QObject
import pyqtgraph as pg
import nidaqmx
from nidaqmx.constants import AcquisitionType, Edge, TerminalConfiguration
from nidaqmx.stream_writers import AnalogMultiChannelWriter
from PyQt5.QtCore import QTimer
import threading

class AnalogInputTask(QThread):
    update_plot = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, channel, sample_rate, buffer_size, config='RSE'):
        super().__init__()
        self.channel = channel
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.config = config
        self.task = None
        self.running = False

    def run(self):
        try:
            with nidaqmx.Task() as self.task:
                if self.config == 'RSE':
                    terminal_config = TerminalConfiguration.RSE
                elif self.config == 'NRSE':
                    terminal_config = TerminalConfiguration.NRSE
                elif self.config == 'DIFF':
                    terminal_config = TerminalConfiguration.DIFF
                elif self.config == 'PSEUDO_DIFF':
                    terminal_config = TerminalConfiguration.PSEUDO_DIFF
                else:
                    terminal_config = TerminalConfiguration.DEFAULT
                
                self.task.ai_channels.add_ai_voltage_chan(self.channel, terminal_config=terminal_config, min_val=-10, max_val=10)
                self.task.timing.cfg_samp_clk_timing(rate=self.sample_rate, sample_mode=AcquisitionType.CONTINUOUS)
                
                self.running = True
                self.start_time = time.time()
                self.task.start()
                
                while self.running:
                    data = self.task.read(number_of_samples_per_channel=self.buffer_size)
                    current_time = time.time()
                    if isinstance(data, (list, np.ndarray)):
                        data = np.array(data).flatten()
                        timestamps = np.linspace(current_time - self.start_time - len(data) / self.sample_rate,
                                                 current_time - self.start_time,
                                                 len(data))
                        self.update_plot.emit(timestamps, data)
                    
                    time.sleep(0.01)  # Reduced delay

        except Exception as e:
            print(f"Error in continuous AI sampling for {self.channel}: {str(e)}")

    def stop(self):
        print(f"Stopping AI task for {self.channel}")
        self.running = False
        
class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)

class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()

class OutputThread(QThread):
    update_plot = pyqtSignal(str, np.ndarray, np.ndarray)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, channel, values, period, iterations, sample_rate):
        super().__init__()
        self.channel = channel
        self.values = values
        self.period = period
        self.iterations = iterations
        self.sample_rate = sample_rate

    def run(self):
        try:
            single_period_samples = len(self.values)
            total_samples = single_period_samples * self.iterations
            waveform = np.tile(self.values, self.iterations)
            
            total_duration = self.period * self.iterations
            timestamps = np.linspace(0, total_duration, total_samples)

            with nidaqmx.Task() as task:
                task.ao_channels.add_ao_voltage_chan(self.channel)
                task.timing.cfg_samp_clk_timing(rate=self.sample_rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=total_samples)
                
                task.write(waveform, auto_start=True)
                
                self.update_plot.emit(self.channel, timestamps, waveform)
                
                task.wait_until_done(timeout=total_duration + 5.0)
            
            self.finished.emit()
        except Exception as e:
            self.error_occurred.emit(str(e))

class DAQmxController(QWidget):
    def __init__(self):
        super().__init__()
        self.threadpool = QThreadPool()

        self.ao_channels = ["Dev1/ao0", "Dev1/ao1", "Dev1/ao2", "Dev1/ao3"]

        self.ao_values = [np.array([]) for _ in range(len(self.ao_channels))]
        self.ao_timestamps = [np.array([]) for _ in range(len(self.ao_channels))]
        
        self.ao_period_textboxes = []
        self.ao_iterations_textboxes = []
        self.ao_file_paths = [None] * len(self.ao_channels)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setVisible(False)
        self.uploaded_data = {channel: None for channel in self.ao_channels}
        self.run_buttons = []
        self.ao_preview_plots = {}
        
        self.ai_channels = ["Dev1/ai0", "Dev1/ai1", "Dev1/ai2", "Dev1/ai3"]
        self.ai_tasks = {}
        self.ai_plots = {}
        self.ai_data = {channel: [[], []] for channel in self.ai_channels}  # (timestamps, values)
        self.ai_sampling_rates = [QLineEdit("10000") for _ in self.ai_channels]
        self.ai_curves = {}  # To store plot curves
        self.ai_config_dropdowns = {}
        self.ai_config_dropdowns = {}

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        # Create separate layouts for AO and AI controls
        ao_control_layout = QGridLayout()
        ai_control_layout = QGridLayout()
        
        # Create a horizontal layout to hold AO and AI control layouts side by side
        control_layout = QHBoxLayout()
        control_layout.addLayout(ao_control_layout)
        control_layout.addLayout(ai_control_layout)
        
        # Create a grid layout for all plots
        plot_layout = QGridLayout()
        plot_layout.setHorizontalSpacing(5)
        plot_layout.setVerticalSpacing(5)
        
        self.ao_labels = []
        self.ao_plots = {}
        self.ai_plots = {}

        # Set up AO controls and preview plots
        for i, ao_channel in enumerate(self.ao_channels):
            ao_label = QLabel(f"Analog Output {ao_channel}")
            ao_upload_button = QPushButton("Upload")
            ao_upload_button.clicked.connect(partial(self.upload_ao_values, ao_channel))

            ao_period_label = QLabel("Period (s):")
            ao_period_textbox = QLineEdit("0.005")
            ao_iterations_label = QLabel("Iterations:")
            ao_iterations_textbox = QLineEdit("1")

            ao_set_button = QPushButton("Set")
            ao_set_button.clicked.connect(partial(self.set_ao_settings, ao_channel))

            run_button = QPushButton("Run")
            run_button.clicked.connect(partial(self.run_ao_values, ao_channel))
            run_button.setEnabled(False)
            self.run_buttons.append(run_button)

            ao_reset_button = QPushButton("Reset")
            ao_reset_button.clicked.connect(partial(self.reset_ao_graph, ao_channel))

            ao_control_layout.addWidget(ao_label, i, 0)
            ao_control_layout.addWidget(ao_upload_button, i, 1)
            ao_control_layout.addWidget(ao_period_label, i, 2)
            ao_control_layout.addWidget(ao_period_textbox, i, 3)
            ao_control_layout.addWidget(ao_iterations_label, i, 4)
            ao_control_layout.addWidget(ao_iterations_textbox, i, 5)
            ao_control_layout.addWidget(ao_set_button, i, 6)
            ao_control_layout.addWidget(run_button, i, 7)
            ao_control_layout.addWidget(ao_reset_button, i, 8)

            self.ao_labels.append(ao_label)
            self.ao_period_textboxes.append(ao_period_textbox)
            self.ao_iterations_textboxes.append(ao_iterations_textbox)

            # Create and add AO preview plot
            preview_plot = pg.PlotWidget(title=f"Preview: {ao_channel}")
            preview_plot.setLabel('left', 'Voltage', units='V')
            preview_plot.setLabel('bottom', 'Sample')
            preview_plot.showGrid(x=True, y=True)
            preview_plot.setYRange(-10, 10, padding=0)
            preview_plot.getAxis('left').setTicks([[(v, str(v)) for v in range(-10, 11, 2)]])
            preview_plot.setFixedSize(300, 200)
            self.ao_preview_plots[ao_channel] = preview_plot
            plot_layout.addWidget(preview_plot, i, 0)

            # Create and add AO plot
            plot_widget_ao = pg.PlotWidget(title=f"Analog Output Wave {ao_channel}")
            plot_widget_ao.setLabel('left', 'Voltage', units='V')
            plot_widget_ao.setLabel('bottom', 'Time', units='s')
            plot_widget_ao.showGrid(x=True, y=True)
            plot_widget_ao.setYRange(-10, 10, padding=0)
            plot_widget_ao.getAxis('left').setTicks([[(v, str(v)) for v in range(-10, 11, 2)]])
            plot_widget_ao.setFixedSize(300, 200)
            self.ao_plots[ao_channel] = plot_widget_ao
            plot_layout.addWidget(plot_widget_ao, i, 1)

        # Set up AI controls and plots
        for i, ai_channel in enumerate(self.ai_channels):
            ai_label = QLabel(f"Analog Input {ai_channel}")
            ai_start_button = QPushButton("Start")
            ai_stop_button = QPushButton("Stop")
            ai_start_button.clicked.connect(partial(self.start_ai_task, ai_channel))
            ai_stop_button.clicked.connect(partial(self.stop_ai_task, ai_channel))

            ai_sampling_rate_label = QLabel("Sampling Rate:")
            ai_sampling_rate = self.ai_sampling_rates[i]

            ai_control_layout.addWidget(ai_label, i, 0)
            ai_control_layout.addWidget(ai_sampling_rate_label, i, 1)
            ai_control_layout.addWidget(ai_sampling_rate, i, 2)
            ai_control_layout.addWidget(ai_start_button, i, 3)
            ai_control_layout.addWidget(ai_stop_button, i, 4)

            # Create and add AI plot
            plot_widget_ai = pg.PlotWidget(title=f"Analog Input {ai_channel}")
            plot_widget_ai.setLabel('left', 'Voltage', units='V')
            plot_widget_ai.setLabel('bottom', 'Time', units='s')
            plot_widget_ai.showGrid(x=True, y=True)
            plot_widget_ai.setYRange(5.5, 6.5)  # Set initial y-range around 6V
            plot_widget_ai.setXRange(0, 1)  # Set initial x-range to 1 second
            plot_widget_ai.setFixedSize(300, 200)  # Add this line
            self.ai_plots[ai_channel] = plot_widget_ai
            self.ai_curves[ai_channel] = plot_widget_ai.plot(pen='y')
            plot_layout.addWidget(plot_widget_ai, i, 2)
            
            # In the initUI method, where you set up AI controls
            ai_config_label = QLabel("Input Config:")
            ai_config_dropdown = QComboBox()
            ai_config_dropdown.addItems(["RSE", "NRSE", "Differential", "PSEUDO_DIFF"])
            ai_config_dropdown.setCurrentText("RSE")  # Set default to RSE
            ai_config_dropdown.currentTextChanged.connect(partial(self.change_ai_config, ai_channel))

            ai_control_layout.addWidget(ai_config_label, i, 5)
            ai_control_layout.addWidget(ai_config_dropdown, i, 6)
            
            ai_set_button = QPushButton("Set")
            ai_set_button.clicked.connect(partial(self.set_ai_config, ai_channel))
            ai_control_layout.addWidget(ai_set_button, i, 7)
            
            ai_reset_button = QPushButton("Reset")
            ai_reset_button.clicked.connect(partial(self.reset_ai_graph, ai_channel))
            ai_control_layout.addWidget(ai_reset_button, i, 8)

            # Store the dropdown in a dictionary for later access
            self.ai_config_dropdowns[ai_channel] = ai_config_dropdown

        # Create scroll areas for controls and plots
        control_widget = QWidget()
        control_widget.setLayout(control_layout)
        control_scroll_area = QScrollArea()
        control_scroll_area.setWidget(control_widget)
        control_scroll_area.setWidgetResizable(True)
        control_scroll_area.setFixedHeight(200)

        plot_widget = QWidget()
        plot_widget.setLayout(plot_layout)
        plot_scroll_area = QScrollArea()
        plot_scroll_area.setWidget(plot_widget)
        plot_scroll_area.setWidgetResizable(True)

        main_layout.addWidget(control_scroll_area)
        main_layout.addWidget(plot_scroll_area)
        
        self.setLayout(main_layout)
        self.setWindowTitle('DAQmx Analog IO Controller')
        self.showMaximized()
     
    def calculate_max_sampling_rate(self):
        active_channels = len(self.ai_tasks)
        if active_channels == 1:
            return 600000  # Max for single channel
        else:
            return min(500000 // (active_channels + 1), 600000)  # Respect multi-channel limit
       
    def change_ai_config(self, channel, config):
        if channel in self.ai_tasks:
            self.stop_ai_task(channel)
        
        # Restart the task with the new configuration
        self.start_ai_task(channel, config)

    def reset_ai_graph(self, channel):
        self.ai_data[channel] = [[], []]
        self.ai_curves[channel].setData([], [])
        self.ai_plots[channel].setYRange(-10, 10)
        self.ai_plots[channel].setXRange(0, 6)
    
    def set_ai_config(self, channel):
        config = self.ai_config_dropdowns[channel].currentText()
        if channel in self.ai_tasks:
            self.stop_ai_task(channel)
        self.start_ai_task(channel, config)
    
    def update_all_ai_tasks(self):
        max_rate = self.calculate_max_sampling_rate()
        for channel, task in self.ai_tasks.items():
            if task.isRunning():
                new_rate = min(float(self.ai_sampling_rates[self.ai_channels.index(channel)].text()), max_rate)
                task.sample_rate = new_rate
                task.buffer_size = int(new_rate * 0.1)
                print(f"Updated {channel} to sample rate {new_rate}")
                
    def start_ai_task(self, channel, config=None):
        print(f"Attempting to start AI task for {channel}")
        
        max_rate = self.calculate_max_sampling_rate()
        index = self.ai_channels.index(channel)
        requested_rate = float(self.ai_sampling_rates[index].text())
        sample_rate = min(requested_rate, max_rate)
        
        buffer_size = int(sample_rate * 0.1)  # 0.1 seconds of data

        if config is None:
            config = self.ai_config_dropdowns[channel].currentText()
        
        # Ensure config is a string
        config = str(config)

        task = AnalogInputTask(channel, sample_rate, buffer_size, config)
        task.update_plot.connect(partial(self.update_ai_plot, channel))
        task.start()

        self.ai_tasks[channel] = task
        print(f"Started AI task for {channel} with sample rate {sample_rate} and configuration {config}")
        
        # Update all other running tasks
        self.update_all_ai_tasks()

    def stop_ai_task(self, channel):
        print(f"Attempting to stop AI task for {channel}")
        if channel in self.ai_tasks:
            self.ai_tasks[channel].stop()
            self.ai_tasks[channel].wait()
            del self.ai_tasks[channel]
            print(f"AI task stopped for {channel}")
            QMessageBox.information(self, "Info", f"Stopped continuous sampling for {channel}")
            
            # Update other tasks
            self.update_all_ai_tasks()
        else:
            print(f"No AI task running for {channel}")
            QMessageBox.warning(self, "Warning", f"No continuous sampling running for {channel}")

    def update_ai_plot(self, channel, timestamps, values):
        try:
            if isinstance(timestamps, np.ndarray) and isinstance(values, np.ndarray):
                if len(timestamps) > 0 and len(values) > 0:
                    self.ai_data[channel][0].extend(timestamps)
                    self.ai_data[channel][1].extend(values)
                    
                    # Keep only the last 60000 points (6 seconds at 10kHz)
                    if len(self.ai_data[channel][0]) > 60000:
                        self.ai_data[channel][0] = self.ai_data[channel][0][-60000:]
                        self.ai_data[channel][1] = self.ai_data[channel][1][-60000:]
                    
                    self.ai_curves[channel].setData(self.ai_data[channel][0], self.ai_data[channel][1])
                    
                    # Update axis ranges
                    x_min = max(0, self.ai_data[channel][0][-1] - 6)
                    x_max = self.ai_data[channel][0][-1]
                    self.ai_plots[channel].setXRange(x_min, x_max)
                    y_min, y_max = min(self.ai_data[channel][1]), max(self.ai_data[channel][1])
                    y_range = y_max - y_min
                    self.ai_plots[channel].setYRange(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
                    
                    print(f"Updating plot for {channel}: min={y_min:.3f}, max={y_max:.3f}, len={len(self.ai_data[channel][0])}")
                else:
                    print(f"Received empty data for {channel}")
            else:
                print(f"Unexpected data format for {channel}: timestamps={type(timestamps)}, values={type(values)}")
        except Exception as e:
            print(f"Error updating AI plot: {str(e)}")
        
    def set_ao_value(self, channel, textbox):
        try:
            value = float(textbox.text())
            with nidaqmx.Task() as task:
                task.ao_channels.add_ao_voltage_chan(channel)
                task.write(value)
        except nidaqmx.errors.DaqError as e:
            QMessageBox.critical(self, "DAQ Error", f"Error setting AO value: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error: {str(e)}")

    def sweep_ao_voltage(self, channel, start_voltage, end_voltage, voltage_step, iterations):
        try:
            start = float(start_voltage.text())
            end = float(end_voltage.text())
            step = float(voltage_step.text())
            num_iterations = int(iterations.text())

            if start >= end:
                raise ValueError("Start voltage must be less than end voltage.")
            if step <= 0:
                raise ValueError("Voltage step must be greater than zero.")
            if num_iterations <= 0:
                raise ValueError("Number of iterations must be greater than zero.")

            voltages = np.arange(start, end + step, step)
            waveform = np.tile(voltages, num_iterations)

            with nidaqmx.Task() as task:
                task.ao_channels.add_ao_voltage_chan(channel)
                task.timing.cfg_samp_clk_timing(rate=1000, sample_mode=AcquisitionType.FINITE, samps_per_chan=len(waveform))
                task.write(waveform, auto_start=True)

                index = self.ao_channels.index(channel)
                timestamps = np.linspace(0, len(waveform) / 1000, len(waveform))
                for i in range(len(waveform)):
                    self.ao_values[index].append(waveform[i])
                    self.ao_timestamps[index].append(timestamps[i])
                    self.ao_plots[channel].clear()
                    self.ao_plots[channel].plot(self.ao_timestamps[index], self.ao_values[index])
                    QApplication.processEvents()

                task.wait_until_done()
                task.stop()

        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))
        except nidaqmx.errors.DaqError as e:
            QMessageBox.critical(self, "DAQ Error", f"Error sweeping AO voltage: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error: {str(e)}")

    def set_ao_settings(self, channel):
        try:
            index = self.ao_channels.index(channel)
            period = float(self.ao_period_textboxes[index].text())
            iterations = int(self.ao_iterations_textboxes[index].text())

            if period <= 0:
                raise ValueError("Period must be greater than zero.")
            if iterations <= 0:
                raise ValueError("Iterations must be greater than zero.")

            if self.uploaded_data[channel] is None:
                raise ValueError("No data uploaded for this channel.")

            values = self.uploaded_data[channel]
            samples = len(values)
            t = np.linspace(0, period * iterations, samples * iterations)
            preview_waveform = np.tile(values, iterations)

            self.ao_preview_plots[channel].clear()
            self.ao_preview_plots[channel].plot(t, preview_waveform)
            self.ao_preview_plots[channel].setLabel('bottom', 'Time', units='s')

            QMessageBox.information(self, "Settings Applied", f"Settings applied for {channel}. Check the preview graph.")

        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error setting AO values: {str(e)}")
            
    def upload_ao_values(self, channel):
        try:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(self, "Open Text File", "", "Text Files (*.txt)")
            if file_path:
                worker = Worker(self._load_file, file_path, channel)
                worker.signals.result.connect(self.update_preview)
                worker.signals.error.connect(self.handle_error)
                self.threadpool.start(worker)
        except Exception as e:
            self.handle_error(("Error", e, ""))

    def _load_file(self, file_path, channel):
        with open(file_path, "r") as file:
            values = [float(line.strip()) for line in file.readlines()]
        return channel, values

    def update_preview(self, result):
        channel, values = result
        self.uploaded_data[channel] = values
        index = self.ao_channels.index(channel)
        
        self.ao_preview_plots[channel].clear()
        self.ao_preview_plots[channel].plot(range(len(values)), values)
        
        self.run_buttons[index].setEnabled(True)
    
    def closeEvent(self, event):
        for channel in list(self.ai_tasks.keys()):
            self.stop_ai_task(channel)
        for task in nidaqmx.system.System().tasks:
            try:
                task.close()
            except:
                pass
        event.accept()

    def run_ao_values(self, channel):
        if self.uploaded_data[channel] is None:
            QMessageBox.warning(self, "Warning", "No data uploaded for this channel.")
            return
        
        index = self.ao_channels.index(channel)
        values = self.uploaded_data[channel]
        period = float(self.ao_period_textboxes[index].text())
        iterations = int(self.ao_iterations_textboxes[index].text())

        sample_rate = len(values) / period
        samples_per_channel = len(values) * iterations

        waveform = np.tile(values, iterations)

        try:
            with nidaqmx.Task() as ao_task:
                ao_task.ao_channels.add_ao_voltage_chan(channel)
                ao_task.timing.cfg_samp_clk_timing(rate=sample_rate,
                                                sample_mode=AcquisitionType.FINITE,
                                                samps_per_chan=samples_per_channel)

                ao_writer = AnalogMultiChannelWriter(ao_task.out_stream)

                ao_writer.write_many_sample(waveform.reshape(1, -1))

                start_time = time.perf_counter()
                ao_task.start()

                ao_task.wait_until_done(timeout=period * iterations + 5.0)

                end_time = time.perf_counter()

            timestamps = np.linspace(0, period * iterations, samples_per_channel)

            self.update_plot(channel, timestamps, waveform)

            expected_duration = period * iterations
            actual_duration = end_time - start_time
            timing_error = abs(actual_duration - expected_duration)

            info_message = (f"Waveform output completed for channel {channel}\n"
                            f"Expected duration: {expected_duration:.6f}s\n"
                            f"Actual duration: {actual_duration:.6f}s\n"
                            f"Timing error: {timing_error:.6f}s")
            
            QMessageBox.information(self, "Success", info_message)

        except nidaqmx.errors.DaqError as e:
            QMessageBox.critical(self, "DAQ Error", f"Error during AO operation: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error: {str(e)}")
        
    def update_plots(self, channel, ao_timestamps, ao_values):
        try:
            # Update AO plot
            self.ao_plots[channel].clear()
            self.ao_plots[channel].plot(ao_timestamps, ao_values)
            
            QApplication.processEvents()
        except Exception as e:
            print(f"Error updating plots: {str(e)}")
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_plot(self, channel, timestamps, values):
        self.ao_plots[channel].clear()
        self.ao_plots[channel].plot(timestamps * 1000, values)  # Convert timestamps to milliseconds for display
        self.ao_plots[channel].setLabel('bottom', 'Time', units='ms')
        self.ao_plots[channel].setXRange(0, timestamps[-1] * 1000)  # Ensure full range is visible
        QApplication.processEvents()
        
    def handle_thread_error(self, error_message):
        QMessageBox.critical(self, "Error", f"An error occurred: {error_message}")

    def output_finished(self, channel):
        QMessageBox.information(self, "Success", f"Waveform output completed for channel {channel}")
        
    def handle_error(self, error_info):
        QMessageBox.critical(self, "Error", str(error_info[1]))
                            
    def reset_ao_graph(self, channel):
        index = self.ao_channels.index(channel)
        self.ao_values[index] = np.array([])
        self.ao_timestamps[index] = np.array([])
        self.ao_plots[channel].clear()
    
    def read_ao_value(self, channel, index):
        try:
            with nidaqmx.Task() as task:
                task.ai_channels.add_ai_voltage_chan(channel,min_val=-10,max_val=10)
                value = task.read()
                self.ao_read_labels[index].setText(f"{value[0]:.2f}")
        except nidaqmx.errors.DaqError as e:
            QMessageBox.critical(self, "DAQ Error", f"Error reading AO value: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DAQmxController()
    sys.exit(app.exec_())

# The multichannel not working.
