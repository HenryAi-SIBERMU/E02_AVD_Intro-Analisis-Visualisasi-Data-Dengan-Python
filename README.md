![MasterHead](https://images.squarespace-cdn.com/content/v1/5feb53185d3dab691b47361b/1609930650139-9NRI63XUJ29Y7E9LEA9G/12eca-machine-learning.gif)

# Outline Pertemuan 2 - Pengenalan lanjutan dan Instalasi Tools Analisis Visualisasi Data Dengan PYTHON

### 📖 Refrensi Buku
1. [Machine Learning Yearning](https://youtube.com/live/3dlj5VZEKN8) By Andrew Ng
2. [The Hundred Page Machine Learning Book](https://youtube.com/live/3dlj5VZEKN8) By Andriy Burkov
3. [Learning From Data](https://youtube.com/live/3dlj5VZEKN8) By Yaser S. Abu Mostafa
4. [Pengenalan Konsep Pembelajaran Mesin dan Deep Learning](https://youtube.com/live/3dlj5VZEKN8) By Jan Wira Gotama Putra

### 📋 Preview Pertemuan Perdana
1. [Penegenalan Analisis dan Visualiasi Data menggunakan Python](https://youtube.com/live/3dlj5VZEKN8)

### 🔎 Intermezzo & Funfact
1. **[Sejarah ML, DL, Ai](#sejarah-ml-dl-ai)**
2. **[Survey Developer Dunia terhadap Python](#survey-developer-dunia-terhadap-python)**
3. **[Dibalik Layar ChatGPT](#dibalik-layar-chatGPT)**

### 💡 Materi - Intro Python, Anaconda, Jupyter 101
1. **[Python Mindmap](#free-books)**
2. **[EDA Workflow](#free-books)**
3. **[Python Roadmap](#free-books)**
4. **[Python Terminology](#free-books)**
5. **[Anaconda Navigator](#free-books)**
6. **[Jupyter Notebook](#free-books)**

### 💡 Materi - Tools Installation
1. **[Python](#free-books)**
2. **[Anaconda](#free-books)**
3. **[Jupyter](#free-books)**
4. **[Package & Library](#free-books)**

### 💡 Materi - Experiment REPO #1
1. **[Repo Data Visualisasi Interaktif Dengan Grafana](#free-books)**

--

<h1 align="center">Sejarah ML, DL, Ai</h1>

![Logo](https://www.algotive.ai/hs-fs/hubfs/00%20Blog/02%20Machine%20Learning/timeline.jpg?width=1200&name=timeline.jpg)

### Early 2000an Prestasi-Prestasi Model ML dari perlombaan kini dikembangkan Raksasa teknologi
- IBM dengan IBM Watson mengalahkan Atlet Jepoardy
- Google dengan Google brain dengan Aplha Go Mengalahkan Atlet go

### Lalu Bermunculan Entitas Perusahaan Berfokus Pada Riset Data Analysis Seperti :
- Google Brain
- Alexnet
- OpenAi
- Deepface
- Google Deepmind
- Resnet

<h1 align="center">Survey Developer Dunia terhadap Python</h1>

![Logo](https://www.heliossolutions.co/blog/wp-content/uploads/2020/06/Top-programming-scripting-and-markup-languages.jpg)

Source: https://insights.stackoverflow.com/survey/2021#section-most-popular-technologies-programming-scripting-and-markup-languages

--
--

<h1 align="center">Dibalik Layar ChatGPT</h1>

# ChatGPT Resources

## Context

Quick investigation yang dilakukan oleh (https://gist.github.com/veekaybee/6f8885e9906aa9c5408ebe5c7e870698), mem breakdown teknologi dibelakangya.

##  Model Architecture

<img width="680" src="https://camo.githubusercontent.com/85d00cf9bca67e33c2d1270b51ff1ac01853b26a8d6bb226b711f859d065b4a6/68747470733a2f2f68756767696e67666163652e636f2f64617461736574732f74726c2d696e7465726e616c2d74657374696e672f6578616d706c652d696d616765732f7265736f6c76652f6d61696e2f696d616765732f74726c5f6f766572766965772e706e67">
Source: https://github.com/lvwerra/trl

>ChatGPT adalah model saudara dari **InstructGPT**, yang dilatih untuk mengikuti instruksi dalam prompt dan memberikan respons yang detail. model ini menggunakan **Reinforcement Learning from Human Feedback (RLHF)**, dengan menggunakan metode yang sama seperti InstructGPT, namun dengan sedikit perbedaan dalam pengaturan pengumpulan data

## Training Data

+ The model data is recent as of 2021 and does offline inference (aka it doesn't know anything about, for example, the death of Queen Elizabeth 2). 

<img width="805" alt="Screen Shot 2022-12-08 at 2 49 16 PM" src="https://user-images.githubusercontent.com/3837836/206862175-89a81508-0f8f-49e9-b79a-f8af73bf7fd8.png">

Originally I asked about this on Twitter and didn't come up with much. [My Twitter Thread Question on Training Data](https://twitter.com/vboykis/status/1290030614410702848). But since then, independent researchers have been discussing and verifying the very opaque training data behind the OpenAI models. 

A key component of GPT-3x models are [Books1 and Books2](https://twitter.com/theshawwn/status/1320282151645073408), both of which are shrouded in mystery. Researchers have attempted to recrate the data using OpenBooks1 and 2. 

<img width="475" alt="Screen Shot 2022-12-10 at 2 13 51 PM" src="https://user-images.githubusercontent.com/3837836/206871628-b2a1e151-4585-40cb-aaae-742e1088d442.png">

The model was trained on:

+ [Books1](https://github.com/soskek/bookcorpus/issues/27#issuecomment-716104208) -  also known as BookCorpus. Here's a paper on [BookCorpus](https://arxiv.org/pdf/2105.05241.pdf), which maintains that it's free books scraped from smashwords.com. 
+ Books2 - No one knows exactly what this is, people suspect it's libgen
+ [Common Crawl](https://en.wikipedia.org/wiki/Common_Crawl)
+ [WebText2](https://www.eleuther.ai/projects/owt2/) - an internet dataset created by scraping URLs extracted from Reddit submissions with a minimum score of 3 as a proxy for quality, deduplicated at the document level with [MinHash](https://boringml.com/docs/recsys/minhash/)
+ [What's in MyAI Paper](https://lifearchitect.ai/whats-in-my-ai-paper/), [Source](https://twitter.com/kdamica/status/1600328844753240065) - Detailed dive into these datasets. 


## Model Evaluation

The policy model was [evaluated by humans,](https://github.com/openai/following-instructions-human-feedback/blob/main/model-card.md) 

>InstructGPT is then further fine-tuned on a dataset labeled by human labelers. The labelers comprise a team of about 40 contractors whom we hired through Upwork and ScaleAI. Our aim was to select a group of labelers who were sensitive to the preferences of different demographic groups, and who were good at identifying outputs that were potentially harmful. Thus, we conducted a screening test designed to measure labeler performance on these axes. We selected labelers who performed well on this test. We collaborated closely with the labelers over the course of the project. We had an onboarding process to train labelers on the project, wrote detailed instructions for each task, and answered labeler questions in a shared chat room.
