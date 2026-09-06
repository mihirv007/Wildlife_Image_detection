import { motion, useInView } from 'framer-motion';
import { useRef, useState, useEffect } from 'react';
import './StatsBar.css';

const stats = [
  { value: 90, suffix: '%', label: 'Model Accuracy', icon: '🎯' },
  { value: 5, suffix: '', label: 'Species Supported', icon: '🦁' },
  { value: 1, suffix: 's', prefix: '<', label: 'Detection Speed', icon: '⚡' },
  { value: 10000, suffix: '+', label: 'Images Trained', icon: '🧠' },
];

function AnimatedCounter({ value, suffix = '', prefix = '', inView }) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (!inView) return;
    let start = 0;
    const end = value;
    const duration = 2000;
    const increment = end / (duration / 16);
    const timer = setInterval(() => {
      start += increment;
      if (start >= end) {
        setCount(end);
        clearInterval(timer);
      } else {
        setCount(Math.floor(start));
      }
    }, 16);
    return () => clearInterval(timer);
  }, [inView, value]);

  return (
    <span className="stats__value accent">
      {prefix}{count.toLocaleString()}{suffix}
    </span>
  );
}

export default function StatsBar() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: '-100px' });

  return (
    <section className="stats" ref={ref}>
      <div className="stats__inner glass">
        {stats.map((stat, i) => (
          <motion.div
            key={stat.label}
            className="stats__item"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: i * 0.1 }}
          >
            <span className="stats__icon">{stat.icon}</span>
            <AnimatedCounter
              value={stat.value}
              suffix={stat.suffix}
              prefix={stat.prefix}
              inView={inView}
            />
            <span className="stats__label">{stat.label}</span>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
