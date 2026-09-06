import { motion } from 'framer-motion';
import './SpeciesShowcase.css';

const species = [
  {
    name: 'Cat',
    emoji: '🐱',
    description: 'Domestic cats with diverse breeds and coat patterns',
    color: '#FF6B9D',
  },
  {
    name: 'Dog',
    emoji: '🐕',
    description: 'Loyal companions spanning hundreds of unique breeds',
    color: '#FFA94D',
  },
  {
    name: 'Elephant',
    emoji: '🐘',
    description: 'Majestic giants of the African and Asian landscapes',
    color: '#748FFC',
  },
  {
    name: 'Horse',
    emoji: '🐎',
    description: 'Graceful equines known for speed and elegance',
    color: '#A9E34B',
  },
  {
    name: 'Lion',
    emoji: '🦁',
    description: 'The kings of the savanna with iconic golden manes',
    color: '#FFD43B',
  },
];

const cardVariants = {
  hidden: { opacity: 0, y: 60, scale: 0.9 },
  visible: (i) => ({
    opacity: 1,
    y: 0,
    scale: 1,
    transition: {
      duration: 0.6,
      delay: i * 0.1,
      ease: [0.25, 0.46, 0.45, 0.94],
    },
  }),
};

export default function SpeciesShowcase() {
  return (
    <section className="species section" id="species">
      <motion.span
        className="species__label"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5 }}
      >
        Supported Species
      </motion.span>
      <motion.h2
        className="section-title"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6, delay: 0.1 }}
      >
        Meet the <span className="accent">Wildlife</span>
      </motion.h2>
      <motion.p
        className="section-subtitle"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        Our CNN model is trained to identify these five species with 90% accuracy
      </motion.p>

      <div className="species__grid">
        {species.map((animal, i) => (
          <motion.div
            key={animal.name}
            className="species__card glass glow-border-hover"
            custom={i}
            variants={cardVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: '-50px' }}
            whileHover={{
              y: -12,
              transition: { duration: 0.3 },
            }}
          >
            <div
              className="species__card-glow"
              style={{ background: `radial-gradient(circle at 50% 0%, ${animal.color}20, transparent 70%)` }}
            />
            <motion.div
              className="species__emoji"
              whileHover={{ scale: 1.2, rotate: [0, -10, 10, 0] }}
              transition={{ duration: 0.4 }}
            >
              {animal.emoji}
            </motion.div>
            <h3 className="species__name" style={{ color: animal.color }}>
              {animal.name}
            </h3>
            <p className="species__description">{animal.description}</p>
            <div className="species__tag" style={{ borderColor: `${animal.color}40`, color: animal.color }}>
              Detectable
            </div>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
