import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import './UploadSection.css';

export default function UploadSection() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const f = acceptedFiles[0];
    if (f) {
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setResult(null);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.bmp', '.webp'] },
    maxFiles: 1,
    multiple: false,
  });

  const handleDetect = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch('http://127.0.0.1:5000/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error('Prediction failed');

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError('Could not connect to the detection server. Make sure the Flask backend is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const speciesEmojis = {
    cat: '🐱',
    dog: '🐕',
    elephant: '🐘',
    horse: '🐎',
    lion: '🦁',
  };

  return (
    <section className="upload section" id="upload">
      <motion.span
        className="upload__label"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5 }}
      >
        Try It Now
      </motion.span>
      <motion.h2
        className="section-title"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6, delay: 0.1 }}
      >
        Detect <span className="accent">Species</span>
      </motion.h2>
      <motion.p
        className="section-subtitle"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        Upload a wildlife image and our AI will identify the species instantly
      </motion.p>

      <motion.div
        className="upload__container"
        initial={{ opacity: 0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6, delay: 0.3 }}
      >
        {/* Dropzone */}
        <AnimatePresence mode="wait">
          {!preview ? (
            <motion.div
              key="dropzone"
              {...getRootProps()}
              className={`upload__dropzone glass ${isDragActive ? 'upload__dropzone--active' : ''}`}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              whileHover={{ borderColor: 'rgba(0,255,136,0.5)' }}
            >
              <input {...getInputProps()} />
              <motion.div
                className="upload__dropzone-icon"
                animate={isDragActive ? { scale: 1.2, y: -5 } : { scale: 1, y: 0 }}
                transition={{ type: 'spring', stiffness: 300 }}
              >
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              </motion.div>
              <p className="upload__dropzone-title">
                {isDragActive ? 'Drop your image here!' : 'Drag & drop a wildlife image'}
              </p>
              <p className="upload__dropzone-hint">
                or click to browse • JPG, PNG, WebP supported
              </p>
            </motion.div>
          ) : (
            <motion.div
              key="preview"
              className="upload__preview-area"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ type: 'spring', stiffness: 200 }}
            >
              <div className="upload__preview-wrapper glass">
                <img src={preview} alt="Upload preview" className="upload__preview-image" />
                <button className="upload__preview-remove" onClick={handleReset}>
                  ✕
                </button>
              </div>

              <div className="upload__actions">
                <motion.button
                  className="upload__detect-btn"
                  onClick={handleDetect}
                  disabled={loading}
                  whileHover={!loading ? { scale: 1.05, boxShadow: '0 0 35px rgba(0,255,136,0.3)' } : {}}
                  whileTap={!loading ? { scale: 0.95 } : {}}
                >
                  {loading ? (
                    <span className="upload__loading">
                      <span className="upload__spinner" />
                      Analyzing...
                    </span>
                  ) : (
                    <>
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="11" cy="11" r="8" />
                        <line x1="21" y1="21" x2="16.65" y2="16.65" />
                      </svg>
                      Detect Species
                    </>
                  )}
                </motion.button>
                <motion.button
                  className="upload__reset-btn"
                  onClick={handleReset}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Choose Different
                </motion.button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error */}
        <AnimatePresence>
          {error && (
            <motion.div
              className="upload__error glass"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
            >
              ⚠️ {error}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Result */}
        <AnimatePresence>
          {result && (
            <motion.div
              className="upload__result glass glow-border"
              initial={{ opacity: 0, y: 40, scale: 0.9 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ type: 'spring', stiffness: 200, damping: 20 }}
            >
              <div className="upload__result-header">
                <span className="upload__result-emoji">
                  {speciesEmojis[result.species?.toLowerCase()] || '🔍'}
                </span>
                <div>
                  <h3 className="upload__result-species">{result.species}</h3>
                  <p className="upload__result-subtitle">Species Identified</p>
                </div>
              </div>

              <div className="upload__result-confidence">
                <div className="upload__result-confidence-header">
                  <span>Confidence</span>
                  <span className="accent">{result.confidence}%</span>
                </div>
                <div className="upload__confidence-bar">
                  <motion.div
                    className="upload__confidence-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${result.confidence}%` }}
                    transition={{ duration: 1, delay: 0.3, ease: 'easeOut' }}
                  />
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </section>
  );
}
