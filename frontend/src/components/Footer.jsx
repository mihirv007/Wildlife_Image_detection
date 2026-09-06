import { motion } from 'framer-motion';
import './Footer.css';

const techStack = [
  { label: 'TensorFlow', color: '#FF6F00' },
  { label: 'Flask', color: '#10B981' },
  { label: 'React', color: '#61DAFB' },
  { label: 'Framer Motion', color: '#FF0055' },
  { label: 'CNN', color: '#F59E0B' },
];

export default function Footer() {
  return (
    <motion.footer
      className="footer"
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 0.8 }}
    >
      <div className="footer__inner">
        <div className="footer__top">
          <div className="footer__brand">
            <span className="footer__logo">
              🌿 Wild<span className="accent">Eye</span>
            </span>
            <p className="footer__tagline">
              AI-powered wildlife species detection using deep learning and computer vision.
            </p>
          </div>

          <div className="footer__links-group">
            <h4 className="footer__links-title">Quick Links</h4>
            <a href="#hero" className="footer__link">Home</a>
            <a href="#species" className="footer__link">Species</a>
            <a href="#upload" className="footer__link">Detect</a>
            <a href="#how-it-works" className="footer__link">How It Works</a>
          </div>

          <div className="footer__links-group">
            <h4 className="footer__links-title">Tech Stack</h4>
            <div className="footer__tech-badges">
              {techStack.map((tech) => (
                <span
                  key={tech.label}
                  className="footer__badge"
                  style={{ borderColor: `${tech.color}40`, color: tech.color }}
                >
                  {tech.label}
                </span>
              ))}
            </div>
          </div>
        </div>

        <div className="footer__divider" />

        <div className="footer__bottom">
          <p className="footer__copy">
            © {new Date().getFullYear()} WildEye — Wildlife Image Detection.
            Built with 💚 using TensorFlow & React.
          </p>
          <motion.a
            href="https://github.com/mihirv007/Wildlife_Image_detection"
            target="_blank"
            rel="noopener noreferrer"
            className="footer__github"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
            </svg>
            GitHub
          </motion.a>
        </div>
      </div>
    </motion.footer>
  );
}
