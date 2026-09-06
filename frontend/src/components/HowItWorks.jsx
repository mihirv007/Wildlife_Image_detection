import { motion } from 'framer-motion';
import './HowItWorks.css';

const steps = [
  {
    number: '01',
    title: 'Upload Image',
    description: 'Drag and drop or browse to select any wildlife photograph from your device',
    icon: (
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
        <polyline points="17 8 12 3 7 8" />
        <line x1="12" y1="3" x2="12" y2="15" />
      </svg>
    ),
  },
  {
    number: '02',
    title: 'AI Analyzes',
    description: 'Our Convolutional Neural Network processes the image through multiple layers of analysis',
    icon: (
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 2a4 4 0 014 4c0 1.95-1.4 3.58-3.25 3.93" />
        <path d="M8.24 6.93A4 4 0 0112 2" />
        <path d="M12 22a4 4 0 01-4-4c0-1.95 1.4-3.58 3.25-3.93" />
        <path d="M15.76 17.07A4 4 0 0112 22" />
        <path d="M2 12a4 4 0 014-4c1.95 0 3.58 1.4 3.93 3.25" />
        <path d="M6.93 15.76A4 4 0 012 12" />
        <path d="M22 12a4 4 0 01-4 4c-1.95 0-3.58-1.4-3.93-3.25" />
        <path d="M17.07 8.24A4 4 0 0122 12" />
        <circle cx="12" cy="12" r="2" />
      </svg>
    ),
  },
  {
    number: '03',
    title: 'Get Results',
    description: 'Receive the identified species with confidence score in under one second',
    icon: (
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M22 11.08V12a10 10 0 11-5.93-9.14" />
        <polyline points="22 4 12 14.01 9 11.01" />
      </svg>
    ),
  },
];

export default function HowItWorks() {
  return (
    <section className="how section" id="how-it-works">
      <motion.span
        className="how__label"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5 }}
      >
        Simple Process
      </motion.span>
      <motion.h2
        className="section-title"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6, delay: 0.1 }}
      >
        How It <span className="accent">Works</span>
      </motion.h2>
      <motion.p
        className="section-subtitle"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        Three simple steps to identify any wildlife species
      </motion.p>

      <div className="how__steps">
        {steps.map((step, i) => (
          <motion.div
            key={step.number}
            className="how__step"
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-50px' }}
            transition={{ duration: 0.6, delay: i * 0.2 }}
          >
            <motion.div
              className="how__step-icon glass glow-border"
              whileHover={{
                scale: 1.1,
                boxShadow: '0 0 30px rgba(0,255,136,0.3)',
              }}
            >
              <span className="how__step-icon-svg">{step.icon}</span>
            </motion.div>
            <span className="how__step-number">{step.number}</span>
            <h3 className="how__step-title">{step.title}</h3>
            <p className="how__step-desc">{step.description}</p>

            {/* Arrow connector */}
            {i < steps.length - 1 && (
              <motion.div
                className="how__arrow"
                initial={{ opacity: 0, x: -10 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: i * 0.2 + 0.4 }}
              >
                <svg width="40" height="24" viewBox="0 0 40 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M0 12H36M36 12L26 4M36 12L26 20" stroke="var(--color-accent)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </motion.div>
            )}
          </motion.div>
        ))}
      </div>
    </section>
  );
}
