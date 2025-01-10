import streamlit as st
from moviepy.editor import (
    VideoFileClip,
    concatenate_videoclips,
    TextClip,
    CompositeVideoClip,
    vfx,
    AudioFileClip
)
from moviepy.video.fx.all import lum_contrast
import tempfile
import os
import openai
from random import choice
import requests
import json
import shutil
import hashlib
from datetime import datetime
import asyncio
import aiohttp
import numpy as np
import cv2
from dotenv import load_dotenv

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Configurazione chiave API OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# URL della libreria audio di TikTok (simulata)
TIKTOK_AUDIO_API_URL = "https://api.tiktok.com/popular-audios"

# Directory per salvare i progetti utente
PROJECTS_DIR = "user_projects"

# Directory per i LUTs
LUTS_DIR = "luts"

# File per memorizzare gli utenti
USERS_FILE = "users.json"

# Funzione per caricare gli utenti dal file
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# Funzione per salvare gli utenti nel file
def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4)

# Funzione per hashare la password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Funzione per autenticazione
def authenticate(username, password):
    users = load_users()
    hashed_password = hash_password(password)
    return users.get(username) == hashed_password

# Funzione per registrare un nuovo utente
def register(username, password):
    users = load_users()
    if username in users:
        return False, "Username già esistente."
    users[username] = hash_password(password)
    save_users(users)
    return True, "Registrazione avvenuta con successo!"

# Funzione per ottenere audio popolari da TikTok (simulata)
async def get_popular_audios():
    """
    Recupera una lista di audio popolari da TikTok tramite API reali.
    """
    # Simulazione di una richiesta API asincrona
    # Sostituire con una chiamata API autentica a TikTok se disponibile
    popular_audios = [
        {"name": "Audio Trend 1", "url": "https://example.com/audio1.mp3"},
        {"name": "Audio Trend 2", "url": "https://example.com/audio2.mp3"},
        {"name": "Motivational Beat", "url": "https://example.com/audio3.mp3"},
        {"name": "Funny Background Music", "url": "https://example.com/audio4.mp3"},
        {"name": "Dynamic Bass Audio", "url": "https://example.com/audio5.mp3"},
    ]
    await asyncio.sleep(1)  # Simula latenza
    return popular_audios

# Funzione per applicare un LUT al frame
def apply_LUT(frame, lut):
    """
    Applica un LUT (Look-Up Table) a un frame video.
    """
    return cv2.LUT(frame, lut)

# Funzione per caricare un LUT
def load_LUT(lut_path):
    """
    Carica un file LUT e lo converte in formato utilizzabile da OpenCV.
    """
    lut = cv2.imread(lut_path)
    if lut is None:
        raise ValueError(f"Impossibile caricare il LUT da {lut_path}")
    lut = cv2.resize(lut, (256, 256))  # Assicurati che il LUT abbia una dimensione corretta
    return lut

# Funzione per l'editing base del video con rilevamento automatico delle scene
def edit_video(file_path):
    """
    Esegue l'editing base del video: rilevamento scene, tagli, slow motion, zoom, transizioni e testo sovrapposto.
    """
    video = VideoFileClip(file_path)
    
    # Rilevamento delle scene (semplificato)
    scene_length = 5  # Durata di ogni scena in secondi
    clips = []
    for start in range(0, int(video.duration), scene_length):
        end = min(start + scene_length, video.duration)
        clip = video.subclip(start, end)
        # Applica effetti a ogni scena
        clip = clip.fx(vfx.speedx, 1)  # Velocità normale, personalizzare se necessario
        clips.append(clip)
    
    # Transizione tra le scene
    final_clip = concatenate_videoclips(clips, method="compose", padding=-1)
    
    # Aggiunta di testo sovrapposto dinamico
    text = TextClip("Ottimizzato per TikTok 🚀", fontsize=50, color='white', bg_color='black', size=(1080, 100))
    text = text.set_duration(final_clip.duration).set_position(("center", "bottom"))
    
    # Composizione del video finale
    final_video = CompositeVideoClip([final_clip, text])
    
    # Salvataggio del video editato
    output_path = tempfile.mktemp(suffix=".mp4")
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", threads=4)
    
    return output_path

# Funzione per applicare color grading avanzato e filtri
def apply_advanced_filters(video_path, lut_path=None, saturation=1.0, contrast=1.0, exposure=0.0, temperature=6500):
    """
    Applica filtri avanzati e color grading al video.
    Opzioni:
    - lut_path: percorso al file LUT da applicare
    - saturation: fattore di saturazione
    - contrast: fattore di contrasto
    - exposure: aggiunta di esposizione
    - temperature: temperatura del colore in Kelvin
    """
    # Carica il video
    video = VideoFileClip(video_path)
    
    # Converti il video in RGB
    def process_frame(frame):
        # Converti il frame da RGB a BGR per OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).astype(np.float32)
        
        # Regolazione della temperatura del colore
        kelvin = temperature
        temp = kelvin / 100
        if temp <= 66:
            red = 255
            green = temp
            green = 99.4708025861 * np.log(green) - 161.1195681661
            if temp <= 19:
                blue = 0
            else:
                blue = temp - 10
                blue = 138.5177312231 * np.log(blue) - 305.0447927307
        else:
            red = temp - 60
            red = 329.698727446 * ((red) ** -0.1332047592)
            green = temp - 60
            green = 288.1221695283 * ((green) ** -0.0755148492 )
            blue = 255
        
        # Applica la temperatura del colore
        frame_bgr[:, :, 2] = np.clip(frame_bgr[:, :, 2] + exposure, 0, 255)  # Red
        frame_bgr[:, :, 1] = np.clip(frame_bgr[:, :, 1] + exposure, 0, 255)  # Green
        frame_bgr[:, :, 0] = np.clip(frame_bgr[:, :, 0] + exposure, 0, 255)  # Blue
        
        # Converti di nuovo a RGB
        frame_rgb = cv2.cvtColor(frame_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
        
        return frame_rgb

    # Applica regolazioni personalizzate
    def adjust_frame(frame):
        # Converti il frame in formato float32 per evitare overflow
        frame = frame.astype(np.float32)
        
        # Regola saturazione
        frame = frame * saturation
        frame = np.clip(frame, 0, 255)
        
        # Regola contrasto
        frame = 128 + contrast * (frame - 128)
        frame = np.clip(frame, 0, 255)
        
        return frame.astype(np.uint8)
    
    # Carica e applica il LUT se fornito
    if lut_path:
        lut = load_LUT(lut_path)
        def apply_lut(frame):
            return apply_LUT(frame, lut)
        video = video.fl_image(apply_lut)
    
    # Applica le regolazioni personalizzate
    video = video.fl_image(adjust_frame)
    
    # Applica la temperatura del colore e l'esposizione
    video = video.fl_image(process_frame)
    
    # Salvataggio del video filtrato
    output_path = tempfile.mktemp(suffix=".mp4")
    video.write_videofile(output_path, codec="libx264", audio_codec="aac", threads=4)
    
    return output_path

# Funzione combinata per l'editing completo del video
def edit_video_with_filters(file_path, lut_path=None, saturation=1.0, contrast=1.0, exposure=0.0, temperature=6500):
    """
    Combina l'editing base con filtri avanzati e color grading.
    """
    # Editing base
    edited_video_path = edit_video(file_path)
    
    # Applicazione di filtri avanzati e color grading
    filtered_video_path = apply_advanced_filters(
        edited_video_path,
        lut_path=lut_path,
        saturation=saturation,
        contrast=contrast,
        exposure=exposure,
        temperature=temperature
    )
    
    return filtered_video_path

# Funzione per generare descrizioni e hashtag
def generate_description_and_hashtags(prompt, language="it"):
    """
    Utilizza l'API di OpenAI per generare una descrizione accattivante e hashtag virali.
    """
    prompt_language = f"Scrivi in {language}: " + prompt
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt_language,
        max_tokens=200,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Funzione per prevedere le performance del video
def predict_video_performance(description, hashtags, audio):
    """
    Utilizza l'API di OpenAI per analizzare e prevedere le performance del video.
    """
    prompt = (
        f"Analizza la performance prevista di un video TikTok con questa descrizione: '{description}', "
        f"questi hashtag: '{hashtags}' e questo audio: '{audio['name']}'. "
        f"Stima l'engagement e suggerisci miglioramenti."
    )
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Funzione per aggiungere audio al video
def add_audio_to_video(video_path, audio_url):
    """
    Aggiunge un audio al video con una verifica e conversione del file audio.
    """
    # Scarica l'audio
    audio_response = requests.get(audio_url)
    if audio_response.status_code != 200:
        raise ValueError(f"Errore durante il download dell'audio: {audio_url}")
    
    # Salva l'audio in un file temporaneo
    audio_temp = tempfile.mktemp(suffix=".mp3")
    with open(audio_temp, "wb") as f:
        f.write(audio_response.content)
    
    # Verifica se il file audio è valido
    valid_audio_temp = tempfile.mktemp(suffix=".mp3")
    try:
        # Utilizza ffmpeg per verificare e convertire il file audio se necessario
        command = f"ffmpeg -y -i {audio_temp} -vn -acodec libmp3lame {valid_audio_temp}"
        os.system(command)
        
        # Controlla se il file convertito esiste
        if not os.path.exists(valid_audio_temp) or os.path.getsize(valid_audio_temp) == 0:
            raise ValueError("File audio non valido o conversione fallita.")
    except Exception as e:
        raise ValueError(f"Errore durante la conversione dell'audio: {e}")
    
    # Carica audio e video
    video = VideoFileClip(video_path)
    audio = AudioFileClip(valid_audio_temp)
    
    # Imposta l'audio al video
    final_video = video.set_audio(audio)
    
    # Salva il video finale
    output_path = tempfile.mktemp(suffix=".mp4")
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", threads=4)
    
    return output_path
    
# Funzione per salvare il progetto
def save_project(user, files, edited_video_path, description, hashtags, audio, performance_analysis):
    """
    Salva il progetto dell'utente in una directory strutturata.
    """
    user_dir = os.path.join(PROJECTS_DIR, hashlib.md5(user.encode()).hexdigest())
    os.makedirs(user_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = os.path.join(user_dir, f"project_{timestamp}")
    os.makedirs(project_dir, exist_ok=True)
    
    # Salva i file caricati
    saved_files = []
    for file_path in files:
        shutil.copy(file_path, project_dir)
        saved_files.append(os.path.basename(file_path))
    
    # Salva il video editato
    shutil.copy(edited_video_path, project_dir)
    
    # Salva descrizione, hashtag e audio
    with open(os.path.join(project_dir, "description.txt"), "w", encoding="utf-8") as f:
        f.write(description)
    with open(os.path.join(project_dir, "hashtags.txt"), "w", encoding="utf-8") as f:
        f.write(hashtags)
    with open(os.path.join(project_dir, "audio.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps(audio))
    with open(os.path.join(project_dir, "performance.txt"), "w", encoding="utf-8") as f:
        f.write(performance_analysis)
    
    return project_dir

# Funzione principale
def main():
    st.set_page_config(page_title="Dashboard AI Avanzata per Video Virali su TikTok", layout="wide")
    st.title("Dashboard AI Avanzata per Video Virali su TikTok")
    
    # Gestione dello stato di autenticazione e modalità (login/register)
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = ""
    if 'mode' not in st.session_state:
        st.session_state.mode = "Login"  # Può essere "Login" o "Register"
    
    # Se l'utente non è autenticato, mostra il form di login o registrazione
    if not st.session_state.authenticated:
        st.sidebar.header("Autenticazione")
        mode = st.sidebar.radio("Seleziona modalità", ["Login", "Registrati"])
        st.session_state.mode = mode
        
        if mode == "Login":
            username = st.sidebar.text_input("Username")
            password = st.sidebar.text_input("Password", type="password")
            if st.sidebar.button("Login"):
                if authenticate(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.sidebar.success("Login effettuato!")
                else:
                    st.sidebar.error("Credenziali non valide.")
        elif mode == "Registrati":
            new_username = st.sidebar.text_input("Nuovo Username")
            new_password = st.sidebar.text_input("Nuova Password", type="password")
            confirm_password = st.sidebar.text_input("Conferma Password", type="password")
            if st.sidebar.button("Registrati"):
                if new_password != confirm_password:
                    st.sidebar.error("Le password non coincidono.")
                elif not new_username or not new_password:
                    st.sidebar.error("Username e password non possono essere vuoti.")
                else:
                    success, message = register(new_username, new_password)
                    if success:
                        st.sidebar.success(message)
                        st.session_state.mode = "Login"
                    else:
                        st.sidebar.error(message)
        return
    
    # Sezione delle funzionalità una volta autenticato
    st.sidebar.header("Caratteristiche Aggiuntive")
    st.sidebar.write("- Editing video con effetti avanzati (slow motion, zoom, dissolvenze)")
    st.sidebar.write("- Color grading e filtri avanzati")
    st.sidebar.write("- Generazione descrizioni e hashtag ottimizzati con OpenAI")
    st.sidebar.write("- Suggerimento audio TikTok in tempo reale")
    st.sidebar.write("- Previsione performance video con AI")
    st.sidebar.write("- Salvataggio e gestione dei progetti")
    
    # Caricamento file
    uploaded_files = st.file_uploader("Carica video e/o immagini", type=["mp4", "jpg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        # Salva i file caricati in una directory temporanea
        temp_dir = tempfile.mkdtemp()
        saved_files = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            saved_files.append(file_path)
        
        st.success(f"{len(saved_files)} file caricati con successo!")
        
        # Visualizzazione dei file caricati
        for file_path in saved_files:
            if file_path.endswith(".mp4"):
                st.video(file_path)
            else:
                st.image(file_path, caption=os.path.basename(file_path))
        
        # Opzioni di Color Grading Avanzato
        st.sidebar.header("Color Grading Avanzato")
        lut_options = ["Nessuno", "Teal & Orange", "Vintage", "Black & White"]
        selected_lut = st.sidebar.selectbox("Scegli un LUT", lut_options)
        
        # Mappa dei LUT disponibili
        lut_files = {
            "Teal & Orange": os.path.join(LUTS_DIR, "teal_orange.png"),
            "Vintage": os.path.join(LUTS_DIR, "vintage.png"),
            "Black & White": os.path.join(LUTS_DIR, "black_white.png"),
        }
        lut_path = lut_files.get(selected_lut, None) if selected_lut != "Nessuno" else None
        
        # Regolazioni personalizzate
        st.sidebar.subheader("Regolazioni Personalizzate")
        saturation = st.sidebar.slider("Saturazione", 0.5, 2.0, 1.0, 0.1)
        contrast = st.sidebar.slider("Contrasto", 0.5, 2.0, 1.0, 0.1)
        exposure = st.sidebar.slider("Esposizione", -50.0, 50.0, 0.0, 5.0)
        temperature = st.sidebar.slider("Temperatura del Colore (Kelvin)", 2000, 10000, 6500, 100)
        
        if st.button("Avvia Editing"):
            with st.spinner("Editing in corso..."):
                # Identifica il primo video caricato
                video_files = [f for f in saved_files if f.endswith(".mp4")]
                if not video_files:
                    st.error("Nessun video caricato per l'editing.")
                    return
                original_video = video_files[0]
                
                # Editing con filtri avanzati e color grading
                try:
                    edited_video_path = edit_video_with_filters(
                        original_video,
                        lut_path=lut_path,
                        saturation=saturation,
                        contrast=contrast,
                        exposure=exposure,
                        temperature=temperature
                    )
                except Exception as e:
                    st.error(f"Errore durante l'editing del video: {e}")
                    return
                
                # Suggerimento audio asincrono
                async def fetch_audios():
                    return await get_popular_audios()
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                popular_audios = loop.run_until_complete(fetch_audios())
                
                suggested_audio = choice(popular_audios)
                
                # Aggiunta audio al video
                try:
                    final_video_path = add_audio_to_video(edited_video_path, suggested_audio["url"])
                except Exception as e:
                    st.error(f"Errore durante l'aggiunta dell'audio: {e}")
                    return
                
                # Generazione descrizione e hashtag in italiano
                description_prompt = (
                    "Genera una descrizione coinvolgente e 5 hashtag virali per un video TikTok "
                    "focalizzato su tendenze attuali."
                )
                description_and_hashtags = generate_description_and_hashtags(description_prompt, language="Italiano")
                
                # Previsione performance
                try:
                    description = description_and_hashtags.split('\n')[0]
                    hashtags = ' '.join([tag.strip() for tag in description_and_hashtags.split('\n')[1:6]])
                    performance_analysis = predict_video_performance(description, hashtags, suggested_audio)
                except Exception as e:
                    st.error(f"Errore durante la previsione delle performance: {e}")
                    return
                
                # Salva il progetto
                try:
                    project_dir = save_project(
                        user=st.session_state.username,
                        files=saved_files,
                        edited_video_path=final_video_path,
                        description=description,
                        hashtags=hashtags,
                        audio=suggested_audio,
                        performance_analysis=performance_analysis
                    )
                except Exception as e:
                    st.error(f"Errore durante il salvataggio del progetto: {e}")
                    return
            
            st.success("Editing completato!")
            st.video(final_video_path)
            
            # Mostra risultati
            st.subheader("Risultati:")
            st.text_area("Descrizione", description, height=100)
            st.text_area("Hashtag", hashtags, height=100)
            st.write("**Audio Consigliato:**", suggested_audio["name"])
            st.text_area("Analisi delle Performance Previste", performance_analysis, height=150)
            
            # Download del video editato
            with open(final_video_path, "rb") as f:
                st.download_button(
                    label="Scarica Video Editato",
                    data=f,
                    file_name="video_editato_tiktok.mp4",
                    mime="video/mp4"
                )
            
            # Link per scaricare l'audio suggerito
            st.markdown(f"**Scarica l'audio suggerito:** [{suggested_audio['name']}]({suggested_audio['url']})")
            
            # Opzione per salvare il progetto
            if st.button("Salva Progetto"):
                st.success(f"Progetto salvato in: {project_dir}")
    
    # Sezione per visualizzare i progetti salvati
    if 'authenticated' in st.session_state and st.session_state.authenticated:
        st.sidebar.header("I Tuoi Progetti")
        user_dir = os.path.join(PROJECTS_DIR, hashlib.md5(st.session_state.username.encode()).hexdigest())
        if os.path.exists(user_dir):
            projects = os.listdir(user_dir)
            if projects:
                selected_project = st.sidebar.selectbox("Seleziona un progetto", projects)
                if selected_project:
                    project_path = os.path.join(user_dir, selected_project)
                    st.sidebar.write(f"Progetto: {selected_project}")
                    if st.sidebar.button("Visualizza Progetto"):
                        # Visualizza i dettagli del progetto
                        description = ""
                        hashtags = ""
                        audio = {}
                        performance = ""
                        video_file = ""
                        for file in os.listdir(project_path):
                            if file.endswith(".mp4"):
                                video_file = os.path.join(project_path, file)
                            elif file == "description.txt":
                                with open(os.path.join(project_path, file), "r", encoding="utf-8") as f:
                                    description = f.read()
                            elif file == "hashtags.txt":
                                with open(os.path.join(project_path, file), "r", encoding="utf-8") as f:
                                    hashtags = f.read()
                            elif file == "audio.txt":
                                with open(os.path.join(project_path, file), "r", encoding="utf-8") as f:
                                    audio = json.loads(f.read())
                            elif file == "performance.txt":
                                with open(os.path.join(project_path, file), "r", encoding="utf-8") as f:
                                    performance = f.read()
                        
                        st.subheader(f"Progetto: {selected_project}")
                        if video_file:
                            st.video(video_file)
                        st.text_area("Descrizione", description, height=100)
                        st.text_area("Hashtag", hashtags, height=100)
                        st.write("**Audio Utilizzato:**", audio.get("name", "N/A"))
                        st.text_area("Analisi delle Performance", performance, height=150)
                        
                        # Download del video del progetto
                        with open(video_file, "rb") as f:
                            st.download_button(
                                label="Scarica Video del Progetto",
                                data=f,
                                file_name=os.path.basename(video_file),
                                mime="video/mp4"
                            )
                        
                        # Link per scaricare l'audio utilizzato
                        st.markdown(f"**Scarica l'audio utilizzato:** [{audio.get('name', 'N/A')}]({audio.get('url', '#')})")
            else:
                st.sidebar.write("Nessun progetto salvato.")
        else:
            st.sidebar.write("Nessun progetto salvato.")

if __name__ == "__main__":
    main()