import { motion, useScroll, useTransform } from 'framer-motion';
import { useRef } from 'react';
import './HeroSection.css';

const titleWords = ['Wildlife', 'Image', 'Detection'];

export default function HeroSection() {
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ['start start', 'end start'],
  });

  const bgY = useTransform(scrollYProgress, [0, 1], ['0%', '30%']);
  const overlayOpacity = useTransform(scrollYProgress, [0, 0.5], [0.4, 0.85]);
  const contentY = useTransform(scrollYProgress, [0, 1], ['0%', '20%']);
  const contentOpacity = useTransform(scrollYProgress, [0, 0.6], [1, 0]);

  const particles = Array.from({ length: 30 }, (_, i) => ({
    id: i,
    x: Math.random() * 100,
    y: Math.random() * 100,
    size: Math.random() * 4 + 1,
    duration: Math.random() * 6 + 4,
    delay: Math.random() * 4,
  }));

  return (
    <section className="hero" id="hero" ref={ref}>
      {/* Parallax Background */}
      <motion.div className="hero__bg" style={{ y: bgY }}>
        <img src="/hero-bg.jpg" alt="" aria-hidden="true" />
      </motion.div>

      {/* Dark gradient overlay */}
      <motion.div className="hero__overlay" style={{ opacity: overlayOpacity }} />

      {/* Floating Particles */}
      <div className="hero__particles" aria-hidden="true">
        {particles.map((p) => (
          <motion.span
            key={p.id}
            className="hero__particle"
            style={{
              left: `${p.x}%`,
              top: `${p.y}%`,
              width: p.size,
              height: p.size,
            }}
            animate={{
              y: [0, -30, 0],
              opacity: [0, 0.8, 0],
            }}
            transition={{
              duration: p.duration,
              repeat: Infinity,
              delay: p.delay,
              ease: 'easeInOut',
            }}
          />
        ))}
      </div>

      {/* Content */}
      <motion.div className="hero__content" style={{ y: contentY, opacity: contentOpacity }}>

        {/* Title with staggered word animation */}
        <h1 className="hero__title">
          {titleWords.map((word, i) => (
            <motion.span
              key={word}
              className={`hero__title-word ${i === 0 ? 'accent' : ''}`}
              initial={{ opacity: 0, y: 50, rotateX: -30 }}
              animate={{ opacity: 1, y: 0, rotateX: 0 }}
              transition={{
                duration: 0.7,
                delay: 0.3 + i * 0.15,
                ease: [0.25, 0.46, 0.45, 0.94],
              }}
            >
              {word}{' '}
            </motion.span>
          ))}
        </h1>

        {/* Subtitle */}
        <motion.p
          className="hero__subtitle"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.9 }}
        >
          Identify animal species in real-time using deep learning and computer vision.
          Upload any wildlife image and let our CNN model do the rest.
        </motion.p>

        {/* CTA Buttons */}
        <motion.div
          className="hero__cta-group"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1.1 }}
        >
          <motion.a
            href="#upload"
            className="hero__cta hero__cta--primary"
            whileHover={{
              scale: 1.05,
              boxShadow: '0 0 40px rgba(0,255,136,0.35)',
            }}
            whileTap={{ scale: 0.95 }}
          >
            <span>Start Detecting</span>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </motion.a>

          <motion.a
            href="#how-it-works"
            className="hero__cta hero__cta--secondary"
            whileHover={{
              scale: 1.05,
              borderColor: 'rgba(0,255,136,0.5)',
            }}
            whileTap={{ scale: 0.95 }}
          >
            Learn More
          </motion.a>
          </motion.div>

        {/* Stats preview */}
        <motion.div
          className="hero__stats-preview"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 1.4 }}
        >
          <div className="hero__stat-item">
            <span className="hero__stat-value accent">90%</span>
            <span className="hero__stat-label">Accuracy</span>
          </div>
          <div className="hero__stat-divider" />
          <div className="hero__stat-item">
            <span className="hero__stat-value accent">5</span>
            <span className="hero__stat-label">Species</span>
          </div>
          <div className="hero__stat-divider" />
          <div className="hero__stat-item">
            <span className="hero__stat-value accent">&lt;1s</span>
            <span className="hero__stat-label">Detection</span>
          </div>
        </motion.div>
      </motion.div>


    </section>
  );
}
