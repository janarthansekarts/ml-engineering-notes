import React from 'react';
import Layout from '@theme/Layout';
import styles from './index.module.css';

const sections = [
  { title: 'ML Engineering Fundamentals', count: 12, path: '/docs/ml-engineering-fundamentals/ml-engineering-vs-data-science', emoji: '🏗️' },
  { title: 'MLOps Fundamentals', count: 12, path: '/docs/mlops-fundamentals/what-is-mlops', emoji: '⚙️' },
  { title: 'Model Training Infrastructure', count: 12, path: '/docs/model-training-infrastructure/gpu-computing-fundamentals', emoji: '🖥️' },
  { title: 'Model Serving', count: 12, path: '/docs/model-serving/serving-fundamentals', emoji: '🚀' },
  { title: 'LLM Operations', count: 12, path: '/docs/llm-operations/llm-landscape-2025', emoji: '🤖' },
  { title: 'AI Agents', count: 12, path: '/docs/ai-agents/ai-agent-fundamentals', emoji: '🧠' },
  { title: 'Feature Engineering', count: 12, path: '/docs/feature-engineering/feature-engineering-principles', emoji: '🧪' },
  { title: 'Model Monitoring', count: 12, path: '/docs/model-monitoring/monitoring-fundamentals', emoji: '📊' },
  { title: 'ML Platform Engineering', count: 12, path: '/docs/ml-platform-engineering/ml-platform-architecture', emoji: '🏛️' },
  { title: 'ML Testing', count: 12, path: '/docs/ml-testing/ml-testing-fundamentals', emoji: '✅' },
  { title: 'Production ML Patterns', count: 12, path: '/docs/production-ml-patterns/ml-design-patterns', emoji: '🎯' },
  { title: 'Responsible AI', count: 12, path: '/docs/responsible-ai/responsible-ai-principles', emoji: '⚖️' },
  { title: 'ML on Cloud', count: 12, path: '/docs/ml-on-cloud/aws-sagemaker', emoji: '☁️' },
  { title: 'Emerging ML Engineering', count: 12, path: '/docs/emerging-ml-engineering/foundation-models', emoji: '🔮' },
];

const totalLessons = sections.reduce((sum, s) => sum + s.count, 0);

export default function Home() {
  return (
    <Layout title="ML Engineering Notes" description="168 in-depth lessons on machine learning engineering — MLOps, model serving, LLMOps, AI agents, and production ML systems.">
      <main className={styles.main}>
        <section className={styles.hero}>
          <h1 className={styles.heroTitle}>ML Engineering Notes</h1>
          <p className={styles.heroSubtitle}>
            {totalLessons} in-depth lessons covering MLOps, model training infrastructure, serving,
            LLM operations, AI agents, feature engineering, monitoring, and production ML patterns.
          </p>
          <p className={styles.heroMeta}>Deep-dive reference • Not surface-level tutorials</p>
        </section>

        <section className={styles.sections}>
          <h2 className={styles.sectionsTitle}>All Sections</h2>
          <div className={styles.sectionsGrid}>
            {sections.map((section) => (
              <a key={section.title} href={section.path} className={styles.sectionCard}>
                <span className={styles.sectionEmoji}>{section.emoji}</span>
                <h3 className={styles.sectionName}>{section.title}</h3>
                <span className={styles.sectionCount}>{section.count} lessons</span>
              </a>
            ))}
          </div>
        </section>

        <section className={styles.crossLinks}>
          <h2 className={styles.crossLinksTitle}>Explore My Other Notes</h2>
          <div className={styles.crossLinksGrid}>
            <a className={styles.crossLinkCard} href="https://data-engineering-notes.janarthansekarts.workers.dev/">Data Engineering</a>
            <a className={styles.crossLinkCard} href="https://kubernetes-deep-dive.janarthansekarts.workers.dev/">Kubernetes Deep Dive</a>
            <a className={styles.crossLinkCard} href="https://python-mastery-notes.janarthansekarts.workers.dev/">Python Mastery</a>
            <a className={styles.crossLinkCard} href="https://observability-notes.janarthansekarts.workers.dev/">Observability</a>
            <a className={styles.crossLinkCard} href="https://platform-engineering-notes.janarthansekarts.workers.dev/">Platform Engineering</a>
            <a className={styles.crossLinkCard} href="https://system-design-notes.janarthansekarts.workers.dev/">System Design</a>
            <a className={styles.crossLinkCard} href="https://software-architecture-notes.janarthansekarts.workers.dev/">Software Architecture</a>
            <a className={styles.crossLinkCard} href="https://cloud-notes.janarthansekarts.workers.dev/">Cloud Engineering</a>
          </div>
        </section>
      </main>
    </Layout>
  );
}
