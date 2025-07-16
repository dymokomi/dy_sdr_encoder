#!/usr/bin/env python3
"""Example 3: Interactive QT visualization of SDR word representations.

This example creates a graphical application where you can:
- Load a trained encoder from disk
- Type a word in the input field
- See its 2D sparse distributed representation as a grid
- Active bits are shown in one color, inactive bits in another
"""

import sys
import numpy as np
from pathlib import Path

# Add the source directory to Python path so we can import dy_sdr_encoder
# This makes the example self-contained without requiring package installation
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
        QWidget, QLabel, QLineEdit, QPushButton, QFileDialog,
        QMessageBox, QFrame, QTextEdit, QGroupBox, QGridLayout
    )
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtGui import QFont, QPainter, QColor, QPen, QBrush
except ImportError:
    print("Error: PySide6 is required for this example.")
    print("Install it with: pip install PySide6")
    sys.exit(1)

from dy_sdr_encoder import Encoder


class SDRGridWidget(QWidget):
    """Custom widget to display SDR as a 2D grid of colored squares."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_sdr = None
        self.previous_sdr = None
        self.cell_size = 8  # Size of each cell in pixels
        self.setMinimumSize(400, 400)
        
    def set_sdr(self, sdr_array, keep_previous=True):
        """Set the SDR to display.
        
        Args:
            sdr_array: New SDR to display
            keep_previous: If True, keep the previous SDR for overlap visualization
        """
        if sdr_array is not None:
            if keep_previous and self.current_sdr is not None:
                self.previous_sdr = self.current_sdr.copy()
            elif not keep_previous:
                self.previous_sdr = None
                
            self.current_sdr = sdr_array.copy()
        else:
            self.current_sdr = None
            self.previous_sdr = None
        self.update()  # Trigger repaint
        
    def paintEvent(self, event):
        """Paint the SDR grid with overlap visualization."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(255, 255, 255))  # White background
        
        if self.current_sdr is None:
            # Show placeholder text
            painter.setPen(QColor(128, 128, 128))
            painter.drawText(self.rect(), Qt.AlignCenter, "No word selected")
            return
            
        height, width = self.current_sdr.shape
        
        # Calculate cell size to fit widget
        available_width = self.width() - 20
        available_height = self.height() - 20
        cell_w = available_width // width
        cell_h = available_height // height
        cell_size = min(cell_w, cell_h, 12)  # Max cell size of 12
        
        if cell_size < 1:
            cell_size = 1
            
        # Calculate grid position (centered)
        grid_width = width * cell_size
        grid_height = height * cell_size
        start_x = (self.width() - grid_width) // 2
        start_y = (self.height() - grid_height) // 2
        
        # Draw grid
        painter.setPen(QPen(QColor(200, 200, 200), 1))  # Light gray borders
        
        for i in range(height):
            for j in range(width):
                x = start_x + j * cell_size
                y = start_y + i * cell_size
                
                current_active = self.current_sdr[i, j]
                previous_active = self.previous_sdr[i, j] if self.previous_sdr is not None else False
                
                # Choose color based on bit states
                if current_active and previous_active:
                    # Overlap: both current and previous active - GREEN
                    color = QColor(0, 150, 0)  # Green
                elif current_active and not previous_active:
                    # Current only - BLUE
                    color = QColor(0, 100, 200)  # Blue
                elif not current_active and previous_active:
                    # Previous only - DARK GRAY
                    color = QColor(200, 200, 200)  # Dark gray
                else:
                    # Neither active - LIGHT GRAY
                    color = QColor(240, 240, 240)  # Light gray
                    
                painter.fillRect(x, y, cell_size, cell_size, color)
                
                # Draw border
                painter.drawRect(x, y, cell_size, cell_size)


class EncoderVisualizerApp(QMainWindow):
    """Main application window for SDR visualization."""
    
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("DY SDR Encoder - Word Visualization")
        self.setGeometry(100, 100, 800, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("SDR Word Visualization")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Encoder loading section
        encoder_group = QGroupBox("Load Encoder")
        encoder_layout = QHBoxLayout(encoder_group)
        
        self.encoder_path_label = QLabel("No encoder loaded")
        self.encoder_path_label.setStyleSheet("color: gray;")
        encoder_layout.addWidget(self.encoder_path_label)
        
        load_button = QPushButton("Browse...")
        load_button.clicked.connect(self.load_encoder)
        encoder_layout.addWidget(load_button)
        
        # Quick load buttons for examples
        example1_button = QPushButton("Load Example 1")
        example1_button.clicked.connect(lambda: self.load_example_encoder("embedded_trained_encoder.npz"))
        encoder_layout.addWidget(example1_button)
        
        example2_button = QPushButton("Load Example 2")
        example2_button.clicked.connect(lambda: self.load_example_encoder("file_trained_encoder.npz"))
        encoder_layout.addWidget(example2_button)
        
        layout.addWidget(encoder_group)
        
        # Encoder info section
        self.info_label = QLabel("Load an encoder to begin")
        self.info_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border: 1px solid #ccc;")
        layout.addWidget(self.info_label)
        
        # Word input section
        input_group = QGroupBox("Word Input")
        input_layout = QHBoxLayout(input_group)
        
        input_layout.addWidget(QLabel("Enter word:"))
        
        self.word_input = QLineEdit()
        self.word_input.setPlaceholderText("Type a word and press Enter...")
        self.word_input.returnPressed.connect(self.visualize_word)
        self.word_input.setEnabled(False)
        input_layout.addWidget(self.word_input)
        
        visualize_button = QPushButton("Visualize")
        visualize_button.clicked.connect(self.visualize_word)
        visualize_button.setEnabled(False)
        self.visualize_button = visualize_button
        input_layout.addWidget(visualize_button)
        
        clear_button = QPushButton("Clear Previous")
        clear_button.clicked.connect(self.clear_previous)
        clear_button.setEnabled(False)
        self.clear_button = clear_button
        input_layout.addWidget(clear_button)
        
        layout.addWidget(input_group)
        
        # SDR visualization
        viz_group = QGroupBox("Sparse Distributed Representation")
        viz_layout = QVBoxLayout(viz_group)
        
        # Word being displayed
        self.current_word_label = QLabel("No word selected")
        self.current_word_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.current_word_label.setAlignment(Qt.AlignCenter)
        viz_layout.addWidget(self.current_word_label)
        
        # Grid widget
        self.grid_widget = SDRGridWidget()
        viz_layout.addWidget(self.grid_widget)
        
        # Statistics
        self.stats_label = QLabel("")
        self.stats_label.setAlignment(Qt.AlignCenter)
        viz_layout.addWidget(self.stats_label)
        
        # Color legend
        legend_frame = QFrame()
        legend_layout = QHBoxLayout(legend_frame)
        legend_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create color legend labels
        legend_layout.addWidget(QLabel("Legend:"))
        
        # Green - overlap
        green_label = QLabel("🟩 Overlap")
        green_label.setStyleSheet("color: #009600; font-weight: bold;")
        legend_layout.addWidget(green_label)
        
        # Blue - current only  
        blue_label = QLabel("🟦 Current word")
        blue_label.setStyleSheet("color: #0064c8; font-weight: bold;")
        legend_layout.addWidget(blue_label)
        
        # Dark gray - previous only
        gray_label = QLabel("⬛ Previous word")
        gray_label.setStyleSheet("color: #787878; font-weight: bold;")
        legend_layout.addWidget(gray_label)
        
        legend_layout.addStretch()  # Push everything to the left
        viz_layout.addWidget(legend_frame)
        
        layout.addWidget(viz_group)
        
        # Vocabulary browser
        vocab_group = QGroupBox("Vocabulary Browser")
        vocab_layout = QVBoxLayout(vocab_group)
        
        self.vocab_display = QTextEdit()
        self.vocab_display.setMaximumHeight(100)
        self.vocab_display.setReadOnly(True)
        self.vocab_display.setPlaceholderText("Load an encoder to see vocabulary...")
        vocab_layout.addWidget(self.vocab_display)
        
        layout.addWidget(vocab_group)
        
    def load_encoder(self):
        """Load encoder from file dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Load Trained Encoder", 
            str(Path("models")), 
            "Numpy files (*.npz);;All files (*)"
        )
        
        if file_path:
            self.load_encoder_from_path(file_path)
            
    def load_example_encoder(self, filename):
        """Load one of the example encoders."""
        file_path = Path("models") / filename
        if file_path.exists():
            self.load_encoder_from_path(str(file_path))
        else:
            QMessageBox.warning(
                self, 
                "File Not Found", 
                f"Example encoder not found: {file_path}\n\n"
                f"Please run the corresponding example script first:\n"
                f"- For embedded encoder: python example1_embedded_training.py\n"
                f"- For file encoder: python example2_file_training.py"
            )
            
    def load_encoder_from_path(self, file_path):
        """Load encoder from specific path."""
        try:
            self.encoder = Encoder.load(file_path)
            
            # Update UI
            self.encoder_path_label.setText(f"Loaded: {Path(file_path).name}")
            self.encoder_path_label.setStyleSheet("color: green;")
            
            # Show encoder info
            info_text = (
                f"Vocabulary: {self.encoder.vocab_size} words | "
                f"Grid: {self.encoder.height}×{self.encoder.width} | "
                f"Sparsity: {self.encoder.sparsity:.3f} | "
                f"Active bits: {self.encoder.active_bits}"
            )
            self.info_label.setText(info_text)
            
            # Enable input
            self.word_input.setEnabled(True)
            self.visualize_button.setEnabled(True)
            self.clear_button.setEnabled(True)
            self.word_input.setFocus()
            
            # Show vocabulary (first 100 words)
            vocab_preview = self.encoder.vocab[:100]
            if len(self.encoder.vocab) > 100:
                vocab_text = ", ".join(vocab_preview) + f"\n... and {len(self.encoder.vocab) - 100} more words"
            else:
                vocab_text = ", ".join(vocab_preview)
            self.vocab_display.setText(vocab_text)
            
            # Clear previous visualization
            self.grid_widget.set_sdr(None, keep_previous=False)
            self.current_word_label.setText("Enter a word to visualize")
            self.stats_label.setText("")
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error Loading Encoder", 
                f"Failed to load encoder from {file_path}:\n\n{str(e)}"
            )
            
    def visualize_word(self):
        """Visualize the entered word."""
        if not self.encoder:
            return
            
        word = self.word_input.text().strip().lower()
        if not word:
            return
            
        if word not in self.encoder:
            QMessageBox.warning(
                self, 
                "Word Not Found", 
                f"The word '{word}' is not in the encoder's vocabulary.\n\n"
                f"Available words: {len(self.encoder.vocab)} total\n"
                f"Try one of: {', '.join(self.encoder.vocab[:10])}"
                f"{', ...' if len(self.encoder.vocab) > 10 else ''}"
            )
            return
            
        # Get SDR and visualize
        sdr = self.encoder[word]
        self.grid_widget.set_sdr(sdr, keep_previous=True)
        
        # Update labels  
        self.current_word_label.setText(f"Word: '{word}'")
        
        # Calculate statistics including overlap
        active_count = np.sum(sdr)
        total_bits = sdr.size
        sparsity_actual = active_count / total_bits
        
        stats_text = f"Active bits: {active_count}/{total_bits} ({sparsity_actual:.3f} sparsity)"
        
        # Add overlap statistics if there's a previous SDR
        if self.grid_widget.previous_sdr is not None:
            previous_active = np.sum(self.grid_widget.previous_sdr)
            overlap_count = np.sum(sdr & self.grid_widget.previous_sdr)
            overlap_percent = (overlap_count / active_count) * 100 if active_count > 0 else 0
            
            stats_text += f" | Overlap: {overlap_count} bits ({overlap_percent:.1f}%)"
        
        self.stats_label.setText(stats_text)
        
        # Clear input for next word
        self.word_input.clear()
        
    def clear_previous(self):
        """Clear the previous SDR from visualization."""
        if self.grid_widget.current_sdr is not None:
            # Keep current SDR but clear previous
            current = self.grid_widget.current_sdr.copy()
            self.grid_widget.set_sdr(current, keep_previous=False)
            
            # Update statistics without overlap info
            active_count = np.sum(current)
            total_bits = current.size
            sparsity_actual = active_count / total_bits
            stats_text = f"Active bits: {active_count}/{total_bits} ({sparsity_actual:.3f} sparsity)"
            self.stats_label.setText(stats_text)


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("DY SDR Encoder Visualizer")
    app.setApplicationVersion("1.0")
    
    # Create and show main window
    window = EncoderVisualizerApp()
    window.show()
    
    # Check if example encoders exist and show helpful message
    models_dir = Path("models")
    if not models_dir.exists() or not any(models_dir.glob("*.npz")):
        QMessageBox.information(
            window,
            "Welcome!",
            "Welcome to the SDR Visualizer!\n\n"
            "To get started:\n"
            "1. Run the training examples to create encoders:\n"
            "   • python example1_embedded_training.py\n"
            "   • python example2_file_training.py\n\n"
            "2. Then use the 'Load Example' buttons to load trained encoders\n"
            "3. Type words to see their representations!"
        )
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 