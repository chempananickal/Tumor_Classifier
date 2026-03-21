# Tumor Classifier Product Architecture Documentation (arc42 Template)

*Version: 1.0 | Status: Draft (2026-03-18) | Authors: Chempananickal James (D876), Leithoff (D...), Savkov (D...)

---

## 1. Introduction and Goals

### 1.1 Requirements Overview

- MRI brain tumor classification supporting both fully local and client-server execution
- Healthcare practitioners send images for remote inference (E2EE, stateless on server)
- Model developers can upload ONNX classifiers via API
- Product is open source, Wiki docs, contribution via PR, bug/features via GitHub Issues
- Communication via mailing list (announcements) and Slack group (collaboration)

### 1.2 Quality Goals

| Goal           | Description                                   | Priority |
|----------------|-------------------------------------------------------|----------|
| Accuracy       | Reliable classification results (>90% in independent validation for every accepted model)| Highest  |
| Security       | E2EE, no unencrypted data stored                                                         | Highest  |
| Privacy        | No persistent patient data                                                               | Highest  |
| Explainability | Heatmaps with Grad-CAM                                                                   | High     |
| Customizability| ONNX model upload                                                                        | Medium   |
| Usability      | Simple UI, clear results                                                                 | High     |
| Open Source    | PR/Issue workflow, easy docs                                                             | High     |
| Performance    | Fast local/server-side inference                                                         | Medium   |


### 1.3 Stakeholders

| Role                         | Contact                | Expectations                          |
|------------------------------|------------------------|---------------------------------------|
| Medical Practitioners        | Clinic emails, Slack   | Reliable tumor classification         |
| Healthcare System Administrators | Direct, Slack      | Deployment, compliance                |
| ML Model Developers          | GitHub, Slack          | Upload/test custom models             |
| Open Source Developers       | GitHub, Slack          | Collaboration, thorough docs/issues   |

---

# 2. Architecture Constraints

## 2.1 Technical Constraints

| Constraint | Explanation |
|---|---|
| Python-centric current codebase | The current project is 100% Python, so migration should preserve leverage of existing model and inference logic where possible. |
| Local mode must use STLite/WASM + ONNX | This constrains local execution technology and requires browser-compatible or WASM-compatible model/runtime packaging. |
| Remote mode must support encrypted request/response flow | The architecture must include client-side encryption and carefully bounded server-side decryption/encryption. |
| Explainability must be preserved | Heatmap generation is a product requirement, not optional functionality. |
| Custom model upload must use a standardized API | Model plugins must conform to a stable inference contract. |
| GitHub is the system of record for collaboration | Wiki, PRs, Issues, and Actions are mandatory workflow components. |
| Server hosting target is Hetzner | Production infrastructure should assume Hetzner VMs, networks, storage, and operational conventions. |

## 2.2 Organizational Constraints

| Constraint | Explanation |
|---|---|
| Open-source project governance | The architecture must remain understandable and contributor-friendly. |
| Contributions via PR only | Architectural change must be reviewable and traceable. |
| Bug reports and feature requests via Issues | Product backlog and defect tracking are GitHub-centered. |
| Communication split across mailing list and Slack | Formal announcements and collaborative discussion occur in separate channels. |

## 2.3 Regulatory / Domain Constraints

| Constraint | Explanation |
|---|---|
| Medical context sensitivity | Even if the product is not itself a certified diagnostic device initially, it operates in a medically sensitive domain and must avoid overstated claims. |
| Protected health information risk | MRI images may be sensitive even without explicit metadata. Privacy by design is mandatory. |
| Auditability expectations | Institutions may expect traceability of model version, runtime version, and configuration for each inference session. |
| Data minimization | Persistent storage of sensitive input data should be avoided wherever possible, especially in remote mode. |

## 2.4 Conventions

| Convention | Explanation |
|---|---|
| Architecture documentation in arc42 format | All major architectural documentation should remain aligned with arc42. |
| Mermaid for diagrams | Diagrams should be text-based and version-controllable. |
| Semantic versioning for releases | Improves traceability of deployments and model compatibility. |
| PR-based review for architecture changes | Major architecture changes should be captured in ADRs and reviewed. |

---

## 3. System Scope and Context

## 3.1 Business Context

The system is a product that mediates between practitioners, administrators, model developers, and the technical infrastructure required to classify brain MRI scans.

### External domain interfaces

| Communication Partner | Inputs to System | Outputs from System |
|---|---|---|
| Medical Practitioner | MRI image, inference mode selection, optional model selection | Predicted class, confidence/probabilities, heatmap, status/errors |
| Healthcare System Administrator | Configuration, deployment policy, model allow-list, access policy | Deployment health, audit metadata, operational alerts |
| ML Model Developer | Model package, metadata, validation manifest | Validation result, registration status, compatibility errors |
| Open-Source Developer | PRs, issues, documentation proposals | Review feedback, merged changes, release notes |
| GitHub Wiki | Documentation content source | Published architecture and user documentation |
| GitHub Issues | Bug/feature submissions | Triage state, discussion, status |
| GitHub Actions | Build/deploy triggers | Build artifacts, deployment status |
| Hetzner Runtime | Host resources | Running inference service, logs, metrics |
| Slack / Mailing List | Collaboration and announcements | Shared decisions, roadmap communication |

## 3.2 Technical Context

The product has two operational contexts:

1. **Local Mode**  
   Browser or local package executes UI and ONNX inference locally using STLite/WASM-compatible components.

2. **Remote Mode**  
   Browser or client app encrypts image payload, sends ciphertext to server, server decrypts ephemerally, runs inference, generates heatmap, encrypts result, and returns ciphertext.

### Mapping of input/output to channels

| Input / Output | Channel | Notes |
|---|---|---|
| Local MRI upload | Browser local file API | No network transmission required |
| Local result rendering | In-process UI rendering | Entirely local |
| Remote MRI upload | HTTPS/TLS with client-side encrypted payload | Payload encrypted before transit |
| Remote classification result | HTTPS/TLS with encrypted response body | Result decrypted client-side |
| Model upload | Authenticated HTTPS API | Restricted to authorized model developers/admins |
| Deployment automation | GitHub Actions to Hetzner over secure deployment channel | Prefer ephemeral credentials or deploy tokens |
| Documentation updates | GitHub Wiki web/git workflow | Version-controlled documentation |
| Collaboration | Slack / mailing list | Outside runtime path |

### Scope boundary

Included in scope:
- UI for image upload and result display
- Local inference runtime
- Remote encrypted inference service
- Heatmap generation
- Model registration and compatibility validation
- Deployment automation
- Community contribution/documentation workflows

Out of scope for the initial product architecture:
- PACS integration
- EHR integration
- Patient identity management
- DICOM archive lifecycle
- Hospital billing workflows
- Regulatory certification process execution itself

---

## 4. Solution Strategy

- Security: All transferred data is E2EE, MRI never persists decrypted on server.
- Two execution modes: browser-based WASM for accessibility, server mode for clinics.
- Model extensibility: API for uploading custom ONNX vision models.
- Simple, explainable UI: Streamlit/STLite; Grad-CAM heatmaps for practitioners.
- Open source/transparent workflow: PRs, Issues, Wiki documentation, Slack community.
- Automated deployments: Server rebuilt after every PR merge using GitHub Actions.

---

# 5. Building Block View

## 5.1 Whitebox Overall System

```mermaid
architecture-beta
    group client(cloud)[Client Side]
    service ui(internet)[Practitioner UI] in client
    service local(lock)[Local Inference Mode] in client
    service localmodels(database)[Local Model Registry] in client
    service remote(lock)[Remote Inference Mode] in client

    group server(cloud)[Server Side]
    service api(server)[Inference API] in server
    service engine(server)[Inference and Heatmap Engine] in server
    service models(database)[Server Model Registry] in server
    service audit(database)[Audit Metadata Store] in server

    group ops(cloud)[Operations]
    service ci(server)[GitHub Actions] in ops
    service wiki(internet)[GitHub Wiki] in ops

    ui:R -- L:local
    local:R -- L:localmodels
    ui:B -- T:remote

    remote:R -- L:api
    api:R -- L:engine
    engine:B -- T:models
    api:B -- T:audit

    ci:L -- R:api
```

#### Motivation
Decouples local/remote inference, enables modular model support, and secures all patient data.

#### Contained Building Blocks
- Frontend (Streamlit/STLite)
- Model Handler API
- ONNX Model handler
- Grad-CAM Visualizer
- E2EE client/server layers
- Hetzner server (stateless inference)
- Model storage (developer uploads)

#### Directory/File Locations
- `frontend/` - UI code
- `backend/` - Server, E2EE, inference
- `models/` - Model management/upload logic
- `.github/workflows/` - Deployment scripts
- `docs/` - Wiki, architecture docs

---

## 6. Runtime View

### Scenario 1: Local Classification

```mermaid
sequenceDiagram
  participant Practitioner
  participant STLite
  participant ONNX
  Practitioner->>STLite: Uploads MRI
  STLite->>ONNX: Runs inference
  ONNX-->>STLite: Tumor class + heatmap (Grad-CAM)
  STLite-->>Practitioner: Displays result
```

### Scenario 2: Remote Classification

```mermaid
sequenceDiagram
  participant Practitioner
  participant Streamlit
  participant E2EE
  participant Server
  participant ONNX
  Practitioner->>Streamlit: Uploads MRI
  Streamlit->>E2EE: Encrypts image
  E2EE->>Server: Secure transfer
  Server->>ONNX: Decrypts + infers (no data persists)
  ONNX-->>Server: Tumor class + heatmap
  Server->>E2EE: Encrypts result
  E2EE->>Streamlit: Secure return
  Streamlit-->>Practitioner: Display result
```

### Scenario 3: Model Upload

```mermaid
sequenceDiagram
  participant ModelDev
  participant Frontend
  participant Server
  participant ModelRepo
  ModelDev->>Frontend: Uploads ONNX model
  Frontend->>Server: Validates and registers
  Server->>ModelRepo: Stores for inference
  ModelRepo-->>Server: Ready
  Server-->>Frontend: Confirmation
```

---

## 7. Deployment View

### Infrastructure Level 1

```mermaid
flowchart TD
    user_device["User Device"]
    stlite_local["STLite WASM (Local)"]
    streamlit_client["Streamlit Client (Remote)"]
    hetzner_server["Hetzner Server"]
    onnx_repo["ONNX Model Repo"]
    gha_runner["GitHub Actions Runner"]

    user_device --> stlite_local
    user_device --> streamlit_client
    streamlit_client --> hetzner_server
    hetzner_server --> onnx_repo
    hetzner_server --> gha_runner
    gha_runner --> hetzner_server
```

**Motivation:**  
Separation of local inference (privacy, offline) and remote (scalability, model expansion). Server rebuilt after each PR for security/compliance.

**Performance Features:**  
- CI/CD ensures server always up-to-date
- No persistent decrypted MRIs
- Scalable server for clinics

---


## 8. Cross-Cutting Concepts

Die in diesem Kapitel beschriebenen Konzepte wirken über mehrere Bausteine hinweg. Sie betreffen nicht nur einzelne Klassen oder Module, sondern prägen den Aufbau der Inferenz, die Nutzung der Oberfläche sowie den Umgang mit Modellartefakten und Laufzeitverhalten.

### 8.1 Datenaufbereitung und Transformationspipeline

Die Verarbeitung eingehender MRT-Bilder folgt einer festen Transformationspipeline. Eingaben werden zunächst von störenden Randartefakten bereinigt, anschließend optional auf den relevanten Gehirnbereich zugeschnitten, auf 224 × 224 Pixel skaliert und danach normalisiert. Für das Training kommt zusätzlich ein zufälliges Corner Masking zum Einsatz, um die Abhängigkeit des Modells von Texteinblendungen, Rändern oder sonstigen Bildecken zu reduzieren. Die Vorverarbeitung ist damit ein fester Bestandteil der Systemlogik.

Architektonisch ist diese Trennung relevant, weil das System die Klassifikation nicht direkt auf Rohdaten ausführt. Zwischen Eingabe und Modell liegt eine definierte und wiederverwendbare Transformationsschicht. Dadurch werden zwei Ziele erreicht: Erstens erhält das Modell konsistente Eingaben; zweitens können Anpassungen an der Datenaufbereitung vorgenommen werden, ohne die Modellimplementierung selbst ändern zu müssen. Das verbessert die Wartbarkeit und Verständlichkeit der Lösung.

### 8.2 Modellbereitstellung und Inferenz

Die Anwendung verwendet für die produktive Inferenz einen gespeicherten Checkpoint unter `models/weights/best.pt`. Im Inferenzpfad wird das Modell geladen, in den Evaluierungsmodus gesetzt und anschließend ohne Gradientenberechnung ausgeführt. Die Vorhersage besteht aus Klassenlabel, Konfidenz und Wahrscheinlichkeitsverteilung über alle vier Klassen. Im Checkpoint kann zusätzlich die tatsächliche Klassenreihenfolge abgelegt werden, damit das Mapping zwischen Modelloutput und fachlicher Klasse stabil bleibt. Dies ist relevant, da im Repository selbst ein früheres Problem mit fehlerhafter Klassenreihenfolge dokumentiert ist.

Für die Laufzeit bedeutet das: Bei identischem Eingabebild, unverändertem Checkpoint und gleicher Ausführungsumgebung ist die Vorhersage deterministisch. Diese Eigenschaft ist für die Testbarkeit und die Fehlersuche wichtig, da Ergebnisse reproduzierbar geprüft werden können. Ergänzend wird bereits bei der Datensatzvorbereitung ein fester Seed verwendet, sodass auch die Aufteilung in Train-, Validierungs- und Testdaten reproduzierbar bleibt.

### 8.3 Ergebnisdarstellung und Nachvollziehbarkeit

Die Ausgabe des Systems beschränkt sich nicht auf ein Klassenlabel. Die Streamlit-Anwendung zeigt zusätzlich die Konfidenz der Vorhersage, die Wahrscheinlichkeiten aller Klassen und eine Grad-CAM-Heatmap. Damit wird die Inferenz um eine visuelle Erklärungskomponente ergänzt. Fachlich ersetzt dies keine medizinische Interpretation, technisch erhöht es aber die Nachvollziehbarkeit der Modellentscheidung.

Die Explainability ist damit als querschnittliches Konzept zu verstehen: Sie betrifft sowohl die Inferenzlogik als auch die Präsentationsschicht. Änderungen am Modell oder an der Hook-Position für Grad-CAM wirken sich direkt auf die Ergebnisdarstellung aus und müssen daher gemeinsam betrachtet werden. 

### 8.4 Bedienung und Nutzungsarten

Die Standardnutzung erfolgt über eine Streamlit-Oberfläche mit Dateiupload für MRT-Bilder. Nach dem Upload werden Bild, Vorhersage und Heatmap unmittelbar angezeigt. Für den normalen Einsatz des vorhandenen Modells ist damit keine Arbeit auf Codeebene notwendig. Diese Form der Bedienung unterstützt den Charakter des Systems als lokal ausführbarer Demonstrator mit niedriger Einstiegshürde.

Davon zu trennen ist die erweiterte Nutzung. Das Training eines eigenen Modells, die Vorbereitung des Datensatzes oder der Austausch von Gewichten erfolgen nicht über die Oberfläche, sondern über Skripte und die Kommandozeile. Für die Dokumentation ist diese Trennung wichtig, weil Bedienbarkeit im Standardfall gegeben ist, erweiterte Nutzung aber weiterhin technisches Vorwissen voraussetzt.

### 8.5 Laufzeitverhalten und Ausführungsumgebung

Das Repository ist auf CPU-Ausführung ausgelegt. In der Umgebungsbeschreibung werden vier oder mehr CPU-Kerne sowie ungefähr 16 GB RAM als praktische Ausgangsbasis genannt; GPU-Unterstützung ist im aktuellen Stand nicht eingerichtet. Damit hängt die wahrgenommene Performance im lokalen Betrieb direkt von der verfügbaren Hardware ab. Die Fachlogik bleibt dabei unverändert.

Wird dieselbe Lösung auf einem dedizierten Server mit fest eingeplanter Rechenleistung betrieben, lässt sich das Laufzeitverhalten planbarer gestalten als auf wechselnden lokalen Endgeräten. Die Architektur des aktuellen Stands enthält dafür jedoch noch keine eigene Serverkomponente; vorgesehen ist zunächst die lokale Ausführung über Streamlit und Python.

### 8.6 Logging, Traceability und Betriebsaspekte

Ein eigenständiges Logging-Konzept ist im aktuellen Repository nicht umgesetzt. Sichtbar ist bislang vor allem Konsolenausgabe im Trainingskontext, etwa über ein konfigurierbares Log-Intervall. Eine persistente Protokollierung von Inferenzanfragen, Vorhersagen, Fehlerfällen oder Laufzeiten findet im derzeitigen Stand nicht statt.

Für eine spätere Erweiterung bietet sich eine klare Trennung zwischen Fachfunktion und Betriebsbeobachtung an. Sinnvoll wäre insbesondere die strukturierte Erfassung von Eingabemetadaten, Modellversion, vorhergesagter Klasse, Konfidenz, Laufzeit und Fehlerfällen. Eine mögliche technische Umsetzung könnte ein leichtgewichtiges Logging oder eine Zwischenspeicherung in einer separaten Komponente wie Redis sein. Diese Erweiterung gehört nicht zum aktuellen Ist-Stand, passt aber in die vorhandene Architektur, da Inferenz, Oberfläche und Modellzugriff bereits voneinander getrennt sind.

### 8.7 Fachliche und regulatorische Grenze des Systems

Die Anwendung enthält selbst einen klaren Hinweis, dass es sich um einen Proof of Concept handelt und nicht um einen Ersatz für professionelle medizinische Diagnose oder Behandlung. Diese Grenze ist nicht nur fachlich, sondern auch architektonisch relevant: Das System ist auf nachvollziehbare lokale Inferenz und Demonstration ausgelegt. Anforderungen wie Auditierbarkeit, regulatorische Absicherung, Datenschutzkonzepte für Patientendaten oder hochverfügbare Bereitstellung sind im aktuellen Stand deshalb noch nicht ausgebaut.

---

## 9. Architecture Decisions

Dieses Kapitel hält die wesentlichen Architekturentscheidungen des aktuellen Systemstands fest. Beschrieben werden Entscheidungen, die sich in der vorhandenen Anwendung, den Trainingsartefakten und der Struktur des Systems wiederfinden. Die Lösung ist auf einen nachvollziehbaren, lokal ausführbaren Demonstrator ausgelegt.

### 9.1 DenseNet121 als Modellbasis

Für die Klassifikation wird DenseNet121 als Backbone verwendet. Der Klassifikationskopf ist auf vier Zielklassen angepasst: Glioma, Meningioma, Pituitary und Negative. Diese Modellwahl passt zur restlichen Struktur der Anwendung, weil sie sich sauber in die bestehende Inferenz einfügt und auch die spätere Visualisierung über Grad-CAM unterstützt.

Die Entscheidung für DenseNet121 hält die Modellseite bewusst überschaubar. Das System braucht kein experimentelles oder besonders großes Modell, sondern eine belastbare Grundlage, die im gegebenen Rahmen gut funktioniert und technisch beherrschbar bleibt. Ein Wechsel auf einen anderen Backbone wäre grundsätzlich möglich, würde aber nicht nur das Training betreffen, sondern auch Teile der Visualisierung und der Modellanbindung.

### 9.2 Inferenz mit festem Checkpoint

Die Anwendung arbeitet mit einem bereits trainierten Modellzustand. Für die Nutzung in der Oberfläche wird der gespeicherte Checkpoint `best.pt` geladen. Ein Training findet in der App selbst nicht statt. Damit bleibt der Nutzungsweg klar: Bild hochladen, Modell laden, Vorhersage berechnen, Ergebnis anzeigen.

Diese Entscheidung hält die Oberfläche kompakt und macht das Verhalten der Anwendung reproduzierbar. Für dieselbe Eingabe und denselben Modellstand entsteht dasselbe Ergebnis. Dies ermöglicht standardisierte Tests und Vergleiche. Der Trainingsprozess bleibt davon getrennt und läuft weiterhin über eigene Skripte.

### 9.3 Klassenreihenfolge wird mit dem Modell gespeichert

Die fachliche Bedeutung der Ausgabewerte hängt davon ab, dass die Reihenfolge der Klassen korrekt bleibt. Deshalb wird die Klassenreihenfolge zusammen mit dem Modellzustand gespeichert und beim Laden wieder übernommen. Dieser Punkt ist für das System essentiell. Eine falsche Zuordnung hierbei würde zu potenziell formal korrekten Ergebnissen führen, die gleichzeitig aber eine hohe Chance auf fachlich falsche Vorhersagen haben.

Mit dieser Entscheidung bleibt die Kopplung zwischen Training und Inferenz an einer kritischen Stelle erhalten. Das Modell gibt nicht nur Wahrscheinlichkeiten aus, sondern diese Wahrscheinlichkeiten werden auch in der richtigen Reihenfolge interpretiert. Dadurch sinkt das Risiko stiller Fehler, die im Betrieb nur schwer auffallen würden.

### 9.4 Streamlit als primärer Zugang

Der vorgesehene Zugang zum System ist die lokale Streamlit-Anwendung. Sie bündelt Upload, Modellinitialisierung, Vorhersage und Ergebnisdarstellung in einer Oberfläche. Damit ist der normale Nutzungspfad bewusst kompakt gehalten. Für die Standardnutzung reicht es aus, ein Bild hochzuladen und die Auswertung direkt in der Oberfläche abzulesen.

Diese Entscheidung passt zum Zweck des Systems. Der Schwerpunkt liegt auf einer direkt nutzbaren Anwendung. Der technische Pflegepfad bleibt davon getrennt.
Es gibt außerdem die Möglichkeit, eigene Modelle für die Klassifizierung zu benutzen. Dieses Feature ermöglicht es, flexibel einsetzbar zu sein und die Vorhersagen auf eigene Anforderungen anzupassen. Für die Änderung des Modells wird Kommandozeilenbenutzung vorausgesetzt.

### 9.5 Grad-CAM ist Teil der Standardausgabe

Die Ausgabe enthält eine Grad-CAM-Heatmap, die die Entscheidung des Modells visuell einordnet. Die Visualisierung ist ein elementarer Teil des normalen Ergebnisbilds der Anwendung. Dadurch wird die Vorhersage für den Nutzer besser nachvollziehbar.

Die Entscheidung hat auch technische Folgen. Die Visualisierung hängt von der Struktur des gewählten Modells ab. Änderungen am Backbone wirken sich deshalb nicht nur auf die Klassifikation, sondern auch auf die Erklärbarkeit der Ausgabe aus.

### 9.6 CPU als Zielumgebung

Die aktuelle Ausführung ist auf CPU-Betrieb ausgelegt. Damit bleibt das Setup einfach und reproduzierbar. Für die vorhandene Abgabe ist das sinnvoll, weil keine spezielle GPU-Umgebung vorausgesetzt werden muss.

Die Laufzeit hängt damit stärker von der verfügbaren Hardware ab als bei einem fest bereitgestellten Server. Für lokale Nutzung ist das akzeptabel. In einer späteren Ausbaustufe könnte dieselbe Fachlogik auch auf eine stabilere Serverumgebung verschoben werden, ohne dass der fachliche Ablauf der Inferenz geändert werden müsste.

---

## 10. Quality Requirements

- Security: E2EE, ephemeral data
- Usability: Streamlined UI, onboarding
- Explainability: Grad-CAM and clear API
- Maintainability: Open source docs, code review
- Extensibility: ONNX plug-ins via API
- Scalability: Automated deployment, cloud server ops

---

## 11. Documentation and Collaboration

- **Documentation:**  
  - GitHub Wiki (Arc42), markdown files
- **Collaboration:**  
  - PRs/Issues on GitHub
  - Announcements via mailing list
  - Slack group for live team discussion

---

## 12. Appendix

### References

- [arc42.org](https://arc42.org)
- [ONNX](https://onnx.ai)
- [Streamlit](https://streamlit.io)
- [Hetzner Cloud](https://www.hetzner.com/cloud)
- [Grad-CAM Explanation](https://arxiv.org/abs/1610.02391)

### Glossary

| Term      | Definition                                     |
|-----------|------------------------------------------------|
| E2EE      | End-to-End Encryption                          |
| ONNX      | Open Neural Network Exchange                   |
| STLite    | Package to run streamlit apps locally with WASM|
| CI/CD     | Continuous Integration/Deployment              |

---
