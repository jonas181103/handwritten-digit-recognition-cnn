# Implementierung eines Convolutional-Neural-Networks  (CNN) zur Klassifikation handschriftlicher Ziffern

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Finished-Finished)]()

---

### 1. Projektübersicht

Dieses Projekt dient der **automatisierten Erkennung und Klassifikation handschriftlicher Ziffern (0-9)**. Es
demonstriert die Anwendung von **Deep Learning** und speziell eines **Convolutional Neural Network (CNN)**.

### 1.1 Technische Kernelemente

* **Architektur:** Convolutional Neural Network (CNN)
* **Datensatz:** MNIST (Modified National Institute of Standards and Technology)
* **Programmiersprache:** Python

### 1.2 Projektmanagement
[Zum Jira Kanban Board](https://handwritten-digit-recognition-cnn.atlassian.net/jira/software/projects/HDRC/boards/1)

---

## 2. Installation und Setup

Dieses Projekt erfordert Python (Version 3.10 oder höher). Alle notwendigen Abhängigkeiten sind in der Datei
`requirements.txt` aufgelistet.

### 2.1 Umgebung einrichten

1. **Repository klonen:**
   ```bash
   git clone https://github.com/jonas181103/handwritten-digit-recognition-cnn
   ```

2. **Virtuelle Umgebung erstellen und aktivieren:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Unter Linux/macOS
   .\venv\Scripts\activate   # Unter Windows
   ```

3. **Abhängigkeiten installieren (aus `requirements.txt`):**
   ```bash
   pip install -r requirements.txt
   ```

---

## 3. Ausführung des Projekts

Die Hauptlogik wird über die zentrale Datei `main.py` gesteuert.

### 3.1 Training und Evaluierung

Um das Modell zu trainieren und anschließend auf dem Testdatensatz zu evaluieren, führen Sie `main.py` aus:

```bash
python main.py
