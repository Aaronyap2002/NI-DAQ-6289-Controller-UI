#Multi channel

import sys
import numpy as np
import time
from functools import partial
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QGridLayout, 
    QScrollArea, QComboBox, QLineEdit, QFileDialog, QMessageBox, QProgressBar, QProgressDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRunnable, QThreadPool, QObject
import pyqtgraph as pg
import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType, Edge
from nidaqmx.stream_writers import AnalogMultiChannelWriter
from nidaqmx.stream_readers import AnalogMultiChannelReader


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
            
            # Calculate total duration based on period and iterations
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
            
class DataProcessingThread(QThread):
    update_progress = pyqtSignal(int)
    update_plot = pyqtSignal(str, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, channel, waveform, total_duration, ai_task, ao_task):
        super().__init__()
        self.channel = channel
        self.waveform = waveform
        self.total_duration = total_duration
        self.stop_event = threading.Event()

    def run(self):
        try:
            with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task:
                # Configure AO task
                ao_task.ao_channels.add_ao_voltage_chan(self.channel)
                ao_task.timing.cfg_samp_clk_timing(rate=len(self.waveform)/self.total_duration, 
                                                   sample_mode=AcquisitionType.FINITE, 
                                                   samps_per_chan=len(self.waveform))

                # Configure AI task
                ai_channel = f"Dev1/ai{self.channel[-1]}"  # Assumes channel format like "Dev1/ao0"
                ai_task.ai_channels.add_ai_voltage_chan(ai_channel)
                ai_task.timing.cfg_samp_clk_timing(rate=len(self.waveform)/self.total_duration, 
                                                   sample_mode=AcquisitionType.FINITE, 
                                                   samps_per_chan=len(self.waveform))

                # Set up synchronization
                ai_task.triggers.start_trigger.cfg_dig_edge_start_trig(f"/{self.channel}/StartTrigger")

                writer = AnalogMultiChannelWriter(ao_task.out_stream)
                reader = AnalogMultiChannelReader(ai_task.in_stream)

                writer.write_many_sample(np.array([self.waveform]))
                
                ao_task.start()
                ai_task.start()

                ao_data = self.waveform
                ai_data = np.zeros((1, len(self.waveform)))
                timestamps = np.linspace(0, self.total_duration, len(self.waveform))

                batch_size = 1000
                for i in range(0, len(self.waveform), batch_size):
                    if self.stop_event.is_set():
                        break
                    end = min(i + batch_size, len(self.waveform))
                    reader.read_many_sample(
                        ai_data[:, i:end], 
                        number_of_samples_per_channel=end - i,
                        timeout=5.0
                    )
                    progress = int((i / len(self.waveform)) * 100)
                    self.update_progress.emit(progress)
                    
                    self.update_plot.emit(self.channel, timestamps[:end], ao_data[:end], timestamps[:end], ai_data[0, :end])

                ao_task.wait_until_done(timeout=self.total_duration + 5.0)
                ai_task.wait_until_done(timeout=self.total_duration + 5.0)
                
                self.update_plot.emit(self.channel, timestamps, ao_data, timestamps, ai_data[0])
                self.update_progress.emit(100)

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.stop_event.set()

    def stop(self):
        self.stop_event.set()

class DAQmxController(QWidget):
    def __init__(self):
        super().__init__()
        self.threadpool = QThreadPool()

        self.ao_channels = ["Dev1/ao0", "Dev1/ao1", "Dev1/ao2", "Dev1/ao3"]
        self.ai_channels = ["Dev1/ai0", "Dev1/ai1", "Dev1/ai2", "Dev1/ai3"]
        self.terminal_configs = ["RSE", "Differential", "Pseudodifferential"]

        self.ao_values = [np.array([]) for _ in range(len(self.ao_channels))]
        self.ai_values = [np.array([]) for _ in range(len(self.ai_channels))]
        self.ao_timestamps = [np.array([]) for _ in range(len(self.ao_channels))]
        self.ai_timestamps = [np.array([]) for _ in range(len(self.ai_channels))]
        
        self.ao_period_textboxes = []
        self.ao_iterations_textboxes = []
        self.ao_file_paths = [None] * len(self.ao_channels)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setVisible(False)
        self.uploaded_data = None
        self.run_buttons = []
        self.ao_preview_plots = {}  # New dictionary for preview plots
        
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        control_layout = QGridLayout()
        control_layout.setHorizontalSpacing(2)
        control_layout.setVerticalSpacing(2)
        
        upload_button = QPushButton("Upload Multichannel Data")
        upload_button.clicked.connect(self.upload_multichannel_data)
        control_layout.addWidget(upload_button, 0, 1)

        # Create a grid layout for the plots
        plot_layout = QGridLayout()
        plot_layout.setHorizontalSpacing(5)
        plot_layout.setVerticalSpacing(5)
        
        self.ao_labels = []
        self.ai_labels = []
        self.ao_plots = {}
        self.ai_plots = {}
        self.ai_terminal_configs = []
        self.ai_sampling_rates = []
        self.ai_min_voltages = []
        self.ai_max_voltages = []

        # Add a single "Run All Channels" button
        self.run_button = QPushButton("Run All Channels")
        self.run_button.clicked.connect(self.run_ao_values)
        self.run_button.setEnabled(False)
        control_layout.addWidget(self.run_button, 0, 9)  # Adjust the row and column as needed
            
        # Create textboxes, labels, upload buttons, period, iterations fields, and set buttons for analog output channels
        for i, ao_channel in enumerate(self.ao_channels):
            ao_label = QLabel(f"Analog Output {ao_channel}")

            ao_period_label = QLabel("Period (s):")
            ao_period_textbox = QLineEdit("0.005")  # Default period of 0.005 seconds (200 Hz)
            ao_iterations_label = QLabel("Iterations:")
            ao_iterations_textbox = QLineEdit("1")  # Default iterations of 1

            ao_set_button = QPushButton("Set")
            ao_set_button.clicked.connect(partial(self.set_ao_settings, ao_channel))

            control_layout.addWidget(ao_label, i, 0)
            control_layout.addWidget(ao_period_label, i, 2)
            control_layout.addWidget(ao_period_textbox, i, 3)
            control_layout.addWidget(ao_iterations_label, i, 4)
            control_layout.addWidget(ao_iterations_textbox, i, 5)
            control_layout.addWidget(ao_set_button, i, 6)
            
            self.ao_labels.append(ao_label)
            self.ao_period_textboxes.append(ao_period_textbox)
            self.ao_iterations_textboxes.append(ao_iterations_textbox)

        # Create labels, read buttons, terminal configuration, sampling rate, and voltage range input fields for analog input channels
        for i, ai_channel in enumerate(self.ai_channels):
            ai_label = QLabel(f"Analog Input {ai_channel}")
            ai_value_label = QLabel("0.0")

            ai_terminal_config = QComboBox()
            ai_terminal_config.addItems(self.terminal_configs)
            self.ai_terminal_configs.append(ai_terminal_config)

            ai_sampling_rate = QLineEdit("1000")
            self.ai_sampling_rates.append(ai_sampling_rate)

            ai_min_voltage = QLineEdit("-10")
            ai_max_voltage = QLineEdit("10")
            self.ai_min_voltages.append(ai_min_voltage)
            self.ai_max_voltages.append(ai_max_voltage)

            control_layout.addWidget(ai_label, i + len(self.ao_channels), 0)
            control_layout.addWidget(QLabel("Terminal Config:"), i + len(self.ao_channels), 2)
            control_layout.addWidget(ai_terminal_config, i + len(self.ao_channels), 3)
            control_layout.addWidget(QLabel("Min Voltage:"), i + len(self.ao_channels), 6)
            control_layout.addWidget(ai_min_voltage, i + len(self.ao_channels), 7)
            control_layout.addWidget(QLabel("Max Voltage:"), i + len(self.ao_channels), 8)
            control_layout.addWidget(ai_max_voltage, i + len(self.ao_channels), 9)

            self.ai_labels.append(ai_value_label)
        
        control_widget = QWidget()
        control_widget.setLayout(control_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidget(control_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedHeight(400)

        main_layout.addWidget(scroll_area)

        # Create plot widgets for each analog output channel
        for i, ao_channel in enumerate(self.ao_channels):
            plot_widget_ao = pg.PlotWidget(title=f"Analog Output Wave {ao_channel}")
            plot_widget_ao.setLabel('left', 'Voltage', units='V')
            plot_widget_ao.setLabel('bottom', 'Time', units='s')
            plot_widget_ao.showGrid(x=True, y=True)
            plot_widget_ao.setYRange(-10, 10, padding=0)
            plot_widget_ao.getAxis('left').setTicks([[(v, str(v)) for v in range(-10, 11, 2)]])
            plot_widget_ao.setFixedSize(300, 200)
            self.ao_plots[ao_channel] = plot_widget_ao
            plot_layout.addWidget(plot_widget_ao, i, 0)
            
            # Create separate preview plot widgets for each analog output channel
            preview_plot = pg.PlotWidget(title=f"Preview: {ao_channel}")
            preview_plot.setLabel('left', 'Voltage', units='V')
            preview_plot.setLabel('bottom', 'Sample')
            preview_plot.showGrid(x=True, y=True)
            preview_plot.setYRange(-10, 10, padding=0)
            preview_plot.getAxis('left').setTicks([[(v, str(v)) for v in range(-10, 11, 2)]])
            preview_plot.setFixedSize(300, 200)
            self.ao_preview_plots[ao_channel] = preview_plot
            plot_layout.addWidget(preview_plot, i,1)

        for i, ai_channel in enumerate(self.ai_channels):
            plot_widget_ai = pg.PlotWidget(title=f"Analog Input Wave {ai_channel}")
            plot_widget_ai.setLabel('left', 'Voltage', units='V')
            plot_widget_ai.setLabel('bottom', 'Time', units='s')
            plot_widget_ai.showGrid(x=True, y=True)
            plot_widget_ai.setYRange(-10, 10, padding=0)
            plot_widget_ai.getAxis('left').setTicks([[(v, str(v)) for v in range(-10, 11, 2)]])
            plot_widget_ai.setFixedSize(300, 200)
            self.ai_plots[ai_channel] = plot_widget_ai
            plot_layout.addWidget(plot_widget_ai,i,2)

        plot_widget = QWidget()
        plot_widget.setLayout(plot_layout)

        plot_scroll_area = QScrollArea()
        plot_scroll_area.setWidget(plot_widget)
        plot_scroll_area.setWidgetResizable(True)

        main_layout.addWidget(plot_scroll_area)
        self.setLayout(main_layout)
        
        self.setWindowTitle('DAQmx Analog IO Controller')
        self.showMaximized()
        # Add reset buttons for analog output graphs
        for i, ao_channel in enumerate(self.ao_channels):
            ao_reset_button = QPushButton("Reset")
            ao_reset_button.clicked.connect(partial(self.reset_ao_graph, ao_channel))
            control_layout.addWidget(ao_reset_button, i, 11)  

        # Add reset buttons for analog input graphs
        for i, ai_channel in enumerate(self.ai_channels):
            ai_reset_button = QPushButton("Reset")
            ai_reset_button.clicked.connect(partial(self.reset_ai_graph, ai_channel))
            control_layout.addWidget(ai_reset_button, i + len(self.ao_channels), 11)
        
        # Add read analog output button and digit display for each channel
        self.ao_read_labels = []
        self.ao_read_labels = []
        for i, ao_channel in enumerate(self.ao_channels):
            ao_read_label = QLabel("0.0")
            self.ao_read_labels.append(ao_read_label)
            control_layout.addWidget(ao_read_label, i, 7)  # Change the column to 7

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
                    QApplication.processEvents()  # Process events to keep the UI responsive

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
            period = float(self.ao_period_textboxes[0].text())
            iterations = int(self.ao_iterations_textboxes[0].text())

            if period <= 0:
                raise ValueError("Period must be greater than zero.")
            if iterations <= 0:
                raise ValueError("Iterations must be greater than zero.")

            if self.uploaded_data is None:
                raise ValueError("No data uploaded.")

            samples = self.uploaded_data.shape[1]
            t = np.linspace(0, period * iterations, samples * iterations)
            preview_waveform = np.tile(self.uploaded_data, (1, iterations))

            # Update the preview plots
            for i, channel in enumerate(self.ao_channels):
                if i < self.uploaded_data.shape[0]:
                    self.ao_preview_plots[channel].clear()
                    self.ao_preview_plots[channel].plot(t, preview_waveform[i])
                    self.ao_preview_plots[channel].setLabel('bottom', 'Time', units='s')

            QMessageBox.information(self, "Settings Applied", "Settings applied for all channels. Check the preview graphs.")

        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error setting AO values: {str(e)}")

    def _load_file(self, file_path, channel):
        with open(file_path, "r") as file:
            values = [float(line.strip()) for line in file.readlines()]
        return channel, values


    
    def closeEvent(self, event):
        # Ensure all tasks are closed and resources are released
        for task in nidaqmx.system.System().tasks:
            try:
                task.close()
            except:
                pass
        event.accept()

    def run_ao_values(self):
        if self.uploaded_data is None:
            QMessageBox.warning(self, "Warning", "No data uploaded.")
            return
        
        try:
            period = float(self.ao_period_textboxes[0].text())
            iterations = int(self.ao_iterations_textboxes[0].text())

            sample_rate = self.uploaded_data.shape[1] / period
            samples_per_channel = self.uploaded_data.shape[1] * iterations

            # Ensure the waveform is contiguous in memory
            waveform = np.ascontiguousarray(np.tile(self.uploaded_data, (1, iterations)))

            with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task:
                # Configure AO and AI tasks for all channels
                for i, (ao_channel, ai_channel) in enumerate(zip(self.ao_channels, self.ai_channels)):
                    ao_task.ao_channels.add_ao_voltage_chan(ao_channel)
                    terminal_config = getattr(TerminalConfiguration, self.ai_terminal_configs[i].currentText())
                    min_val = float(self.ai_min_voltages[i].text())
                    max_val = float(self.ai_max_voltages[i].text())
                    ai_task.ai_channels.add_ai_voltage_chan(ai_channel, terminal_config=terminal_config, min_val=min_val, max_val=max_val)

                ao_task.timing.cfg_samp_clk_timing(rate=sample_rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=samples_per_channel)
                ai_task.timing.cfg_samp_clk_timing(rate=sample_rate, source="ao/SampleClock", sample_mode=AcquisitionType.FINITE, samps_per_chan=samples_per_channel)

                ao_writer = AnalogMultiChannelWriter(ao_task.out_stream)
                ai_reader = AnalogMultiChannelReader(ai_task.in_stream)

                ao_writer.write_many_sample(waveform)

                ai_data = np.zeros((len(self.ai_channels), samples_per_channel), dtype=np.float64)

                ai_task.start()
                start_time = time.perf_counter()
                ao_task.start()

                ai_reader.read_many_sample(ai_data, number_of_samples_per_channel=samples_per_channel, timeout=period * iterations + 5.0)

                end_time = time.perf_counter()

            timestamps = np.linspace(0, period * iterations, samples_per_channel)

            for i, (ao_channel, ai_channel) in enumerate(zip(self.ao_channels, self.ai_channels)):
                self.update_plot(ao_channel, timestamps, waveform[i], timestamps, ai_data[i])

            # Calculate and display timing information
            expected_duration = period * iterations
            actual_duration = end_time - start_time
            timing_error = abs(actual_duration - expected_duration)

            info_message = (f"Multichannel waveform output and input completed\n"
                            f"Expected duration: {expected_duration:.6f}s\n"
                            f"Actual duration: {actual_duration:.6f}s\n"
                            f"Timing error: {timing_error:.6f}s")
            
            QMessageBox.information(self, "Success", info_message)

        except nidaqmx.errors.DaqError as e:
            QMessageBox.critical(self, "DAQ Error", f"Error during AO/AI operation: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error: {str(e)}")
        
    def update_plots(self, channel, ao_timestamps, ao_values, ai_timestamps, ai_values):
        try:
            # Update AO plot
            self.ao_plots[channel].clear()
            self.ao_plots[channel].plot(ao_timestamps, ao_values)
            
            # Update AI plot
            ai_channel = f"Dev1/ai{self.ao_channels.index(channel)}"
            self.ai_plots[ai_channel].clear()
            self.ai_plots[ai_channel].plot(ai_timestamps, ai_values)
            
            QApplication.processEvents()
        except Exception as e:
            print(f"Error updating plots: {str(e)}")
    
    def _output_and_read_values(self, channel, waveform, timestamps, total_duration):
        try:
            with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task:
                # Configure AO task
                ao_task.ao_channels.add_ao_voltage_chan(channel)
                ao_task.timing.cfg_samp_clk_timing(rate=len(waveform)/total_duration, sample_mode=AcquisitionType.FINITE, samps_per_chan=len(waveform))
                
                # Configure AI task
                ai_channel = f"Dev1/ai{self.ao_channels.index(channel)}"
                ai_task.ai_channels.add_ai_voltage_chan(ai_channel)
                ai_task.timing.cfg_samp_clk_timing(rate=len(waveform)/total_duration, sample_mode=AcquisitionType.FINITE, samps_per_chan=len(waveform))
                
                # Set up synchronization
                ai_task.triggers.start_trigger.cfg_dig_edge_start_trig(f"/{channel}/StartTrigger")
                
                # Write AO data
                ao_writer = AnalogMultiChannelWriter(ao_task.out_stream)
                ao_writer.write_many_sample(np.array([waveform]))
                
                # Prepare AI reader
                ai_reader = AnalogMultiChannelReader(ai_task.in_stream)
                ai_data = np.zeros((1, len(waveform)))
                
                # Start tasks
                ai_task.start()
                ao_task.start()
                
                # Read and update plots
                for i in range(0, len(waveform), 1000):
                    if ao_task.is_task_done():
                        break
                    progress = int((i / len(waveform)) * 100)
                    self.signals.progress.emit(progress)
                    
                    ai_reader.read_many_sample(ai_data[:, i:i+1000], number_of_samples_per_channel=min(1000, len(waveform)-i))
                    self.signals.result.emit((timestamps[:i+1000], waveform[:i+1000], ai_data[0, :i+1000]))
                    time.sleep(0.1)  # Adjust this value to control update frequency
                
                ao_task.wait_until_done(timeout=total_duration + 5.0)
                ai_task.wait_until_done(timeout=total_duration + 5.0)
        except Exception as e:
            self.signals.error.emit(str(e))

    def update_ao_plot(self, channel, timestamps, values):
        self.ao_plots[channel].clear()
        self.ao_plots[channel].plot(timestamps * 1000, values)  # Convert timestamps to milliseconds for display
        self.ao_plots[channel].setLabel('bottom', 'Time', units='ms')
        self.ao_plots[channel].setXRange(0, timestamps[-1] * 1000)  # Ensure full range is visible
        QApplication.processEvents()
        
    def handle_thread_error(self, error_message):
        QMessageBox.critical(self, "Error", f"An error occurred: {error_message}")

    def output_finished(self, channel):
        QMessageBox.information(self, "Success", f"Waveform output completed for channel {channel}")
        
    def _output_ao_values(self, channel):
        index = self.ao_channels.index(channel)
        values = self.uploaded_data[channel]
        period = float(self.ao_period_textboxes[index].text())
        iterations = int(self.ao_iterations_textboxes[index].text())

        waveform = np.tile(values, iterations)
        total_duration = period * iterations * len(values)

        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan(channel)
            task.timing.cfg_samp_clk_timing(rate=len(waveform)/total_duration, sample_mode=AcquisitionType.FINITE, samps_per_chan=len(waveform))
            task.write(waveform, auto_start=True)
            task.wait_until_done(timeout=total_duration + 5.0)

    def output_finished(self, channel):
        logging.debug(f"Output finished for channel {channel}")
        QMessageBox.information(self, "Success", f"Output completed for channel {channel}")

    def handle_error(self, error_info):
        QMessageBox.critical(self, "Error", str(error_info[1]))
                
    def output_ao_values(self, channel):
        try:
            index = self.ao_channels.index(channel)
            file_path = self.ao_file_paths[index]
            period = float(self.ao_period_textboxes[index].text())
            iterations = int(self.ao_iterations_textboxes[index].text())

            if file_path is None:
                raise ValueError("No file uploaded for this channel.")

            with open(file_path, "r") as file:
                values = [float(line.strip()) for line in file.readlines()]

            # Prepare the waveform data
            waveform = np.tile(values, iterations)
            total_duration = period * iterations * len(values)

            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Create tasks for both output and input
            with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task:
                # Configure AO task
                ao_task.ao_channels.add_ao_voltage_chan(channel)
                ao_task.timing.cfg_samp_clk_timing(rate=len(waveform)/total_duration, sample_mode=AcquisitionType.FINITE, samps_per_chan=len(waveform))

                # Configure AI task
                ai_channel = f"Dev1/ai{index}"
                ai_task.ai_channels.add_ai_voltage_chan(
                    ai_channel,
                    terminal_config=getattr(TerminalConfiguration, self.ai_terminal_configs[index].currentText()),
                    min_val=float(self.ai_min_voltages[index].text()),
                    max_val=float(self.ai_max_voltages[index].text())
                )
                ai_task.timing.cfg_samp_clk_timing(rate=len(waveform)/total_duration, sample_mode=AcquisitionType.FINITE, samps_per_chan=len(waveform))

                data_thread = DataProcessingThread(channel, waveform, total_duration, ai_task, ao_task)
                data_thread.update_progress.connect(self.progress_bar.setValue)
                data_thread.update_plot.connect(self.update_plot)
                data_thread.finished.connect(lambda: self.progress_bar.setVisible(False))
                data_thread.start()

                # Wait for the thread to finish
                data_thread.wait()

        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))
        except nidaqmx.errors.DaqError as e:
            QMessageBox.critical(self, "DAQ Error", f"Error outputting AO values: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
    
    def handle_thread_error(self, error_message):
        QMessageBox.critical
    
    def cleanup_tasks(self, ao_task, ai_task):
        try:
            ao_task.close()
            ai_task.close()
        except Exception as e:
            print(f"Error during task cleanup: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            
    def update_progress(self, value):
        self.progress_bar.setValue(value)

    
    def update_plot(self, channel, ao_timestamps, ao_values, ai_timestamps, ai_values):
        
        index = self.ao_channels.index(channel)
        
        # Update AO plot
        self.ao_values[index] = ao_values
        self.ao_timestamps[index] = ao_timestamps
        self.ao_plots[channel].clear()
        self.ao_plots[channel].plot(ao_timestamps * 1000, ao_values)  # Convert to milliseconds
        self.ao_plots[channel].setLabel('bottom', 'Time', units='ms')
        self.ao_plots[channel].setXRange(0, ao_timestamps[-1] * 1000)
        
        # Update AI plot
        ai_channel = f"Dev1/ai{index}"
        self.ai_values[index] = ai_values
        self.ai_timestamps[index] = ai_timestamps
        self.ai_plots[ai_channel].clear()
        self.ai_plots[ai_channel].plot(ai_timestamps * 1000, ai_values)  # Convert to milliseconds
        self.ai_plots[ai_channel].setLabel('bottom', 'Time', units='ms')
        self.ai_plots[ai_channel].setXRange(0, ai_timestamps[-1] * 1000)
        
        QApplication.processEvents()

                            
    def reset_ao_graph(self, channel):
        index = self.ao_channels.index(channel)
        self.ao_values[index] = np.array([])
        self.ao_timestamps[index] = np.array([])
        self.ao_plots[channel].clear()

    def reset_ai_graph(self, channel):
        index = self.ai_channels.index(channel)
        self.ai_values[index] = np.array([])
        self.ai_timestamps[index] = np.array([])
        self.ai_plots[channel].clear()
    
    def read_ao_value(self, channel, index):
        try:
            with nidaqmx.Task() as task:
                task.ai_channels.add_ai_voltage_chan(channel)  # Change to ai_channels
                value = task.read()
                self.ao_read_labels[index].setText(f"{value[0]:.2f}")
        except nidaqmx.errors.DaqError as e:
            QMessageBox.critical(self, "DAQ Error", f"Error reading AO value: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error: {str(e)}")
    
    def upload_multichannel_data(self):
        try:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(self, "Open Multichannel Data File", "", "Text Files (*.txt)")
            if file_path:
                worker = Worker(self._load_multichannel_file, file_path)
                worker.signals.result.connect(self.update_multichannel_preview)
                worker.signals.error.connect(self.handle_error)
                self.threadpool.start(worker)
        except Exception as e:
            self.handle_error(("Error", e, ""))

    def _load_multichannel_file(self, file_path):
        data = np.loadtxt(file_path, delimiter='\t')
        return data.T  # Transpose the data so each row represents a channel

    def update_multichannel_preview(self, data):
        self.uploaded_data = data
        for i, channel in enumerate(self.ao_channels):
            if i < data.shape[0]:
                self.ao_preview_plots[channel].clear()
                self.ao_preview_plots[channel].plot(data[i])
        self.run_button.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DAQmxController()
    sys.exit(app.exec_())
