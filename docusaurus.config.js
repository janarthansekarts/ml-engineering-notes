// @ts-nocheck
const { themes: prismThemes } = require('prism-react-renderer');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'ML Engineering Notes',
  tagline: '168 in-depth lessons on machine learning engineering — MLOps, model serving, LLMOps, fine-tuning infrastructure, AI agents, experiment tracking, and production ML systems.',
  favicon: 'img/favicon.svg',

  url: 'https://ml-engineering-notes.janarthansekarts.workers.dev',
  baseUrl: '/',

  organizationName: 'janarthansekarts',
  projectName: 'ml-engineering-notes',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  headTags: [
    {
      tagName: 'meta',
      attributes: { name: 'robots', content: 'noindex, nofollow' },
    },
  ],

  markdown: {
    format: 'md',
  },

  clientModules: [
    require.resolve('./src/contentProtection.js'),
  ],

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  themes: [
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        hashed: true,
        language: ['en'],
        indexBlog: false,
        forceIgnoreNoIndex: true,
        docsDir: '.',
        docsRouteBasePath: '/docs',
      },
    ],
  ],

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: '.',
          include: ['{00,01,02,03,04,05,06,07,08,09,10,11,12,13}-*/**/*.{md,mdx}'],
          exclude: ['**/node_modules/**'],
          routeBasePath: 'docs',
          sidebarPath: './sidebars.js',
          numberPrefixParser: true,
          sidebarCollapsed: true,
          sidebarCollapsible: true,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/social-card.png',
      docs: {
        sidebar: {
          autoCollapseCategories: true,
        },
      },
      navbar: {
        title: 'ML Engineering Notes',
        logo: {
          alt: 'ML Engineering Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'docs',
            position: 'left',
            label: 'All Lessons',
          },
          {
            type: 'dropdown',
            label: 'My Notes',
            position: 'right',
            items: [
              { label: 'PHP & Laravel Notes', href: 'https://php-laravel-core-notes.janarthansekarts.workers.dev/' },
              { label: 'JavaScript Fullstack', href: 'https://javascript-fullstack-notes.janarthansekarts.workers.dev/' },
              { label: 'JavaScript Mastery', href: 'https://javascript-mastery-notes.janarthansekarts.workers.dev/' },
              { label: 'Python Mastery', href: 'https://python-mastery-notes.janarthansekarts.workers.dev/' },
              { label: 'Go Programming', href: 'https://go-programming-notes.janarthansekarts.workers.dev/' },
              { label: 'DSA & Coding Patterns', href: 'https://dsa-coding-notes.janarthansekarts.workers.dev/' },
              { label: 'System Design', href: 'https://system-design-notes.janarthansekarts.workers.dev/' },
              { label: 'Software Architecture', href: 'https://software-architecture-notes.janarthansekarts.workers.dev/' },
              { label: 'API Design', href: 'https://api-design-notes.janarthansekarts.workers.dev/' },
              { label: 'Database Deep Dive', href: 'https://database-deep-dive.janarthansekarts.workers.dev/' },
              { label: 'Cloud Engineering', href: 'https://cloud-notes.janarthansekarts.workers.dev/' },
              { label: 'DevOps & SRE', href: 'https://devops-sre-notes.janarthansekarts.workers.dev/' },
              { label: 'Kubernetes Deep Dive', href: 'https://kubernetes-deep-dive.janarthansekarts.workers.dev/' },
              { label: 'Platform Engineering', href: 'https://platform-engineering-notes.janarthansekarts.workers.dev/' },
              { label: 'Data Engineering', href: 'https://data-engineering-notes.janarthansekarts.workers.dev/' },
              { label: 'Networking Deep Dive', href: 'https://networking-deep-dive.janarthansekarts.workers.dev/' },
              { label: 'Observability', href: 'https://observability-notes.janarthansekarts.workers.dev/' },
              { label: 'Learn AI & ML', href: 'https://learn-ai-ml.janarthansekarts.workers.dev/' },
              { label: 'Security Engineering', href: 'https://security-engineering-notes.janarthansekarts.workers.dev/' },
              { label: 'Blockchain & Web3', href: 'https://blockchain-web3-notes.janarthansekarts.workers.dev/' },
            
            ],
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'ML Foundations',
            items: [
              { label: 'Fundamentals', to: '/docs/ml-engineering-fundamentals/ml-engineering-vs-data-science' },
              { label: 'MLOps', to: '/docs/mlops-fundamentals/what-is-mlops' },
              { label: 'Training Infrastructure', to: '/docs/model-training-infrastructure/gpu-computing-fundamentals' },
              { label: 'Model Serving', to: '/docs/model-serving/serving-fundamentals' },
            ],
          },
          {
            title: 'Advanced ML Engineering',
            items: [
              { label: 'LLM Operations', to: '/docs/llm-operations/llm-landscape-2025' },
              { label: 'AI Agents', to: '/docs/ai-agents/ai-agent-fundamentals' },
              { label: 'ML Platform', to: '/docs/ml-platform-engineering/ml-platform-architecture' },
              { label: 'Production Patterns', to: '/docs/production-ml-patterns/ml-design-patterns' },
            ],
          },
          {
            title: 'Related Projects',
            items: [
              { label: 'Data Engineering', href: 'https://data-engineering-notes.janarthansekarts.workers.dev/' },
              { label: 'Kubernetes Deep Dive', href: 'https://kubernetes-deep-dive.janarthansekarts.workers.dev/' },
              { label: 'Python Mastery', href: 'https://python-mastery-notes.janarthansekarts.workers.dev/' },
            ],
          },
        ],
        copyright: `© ${new Date().getFullYear()} Janarthan Sekar. Deep-dive reference notes.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['python', 'bash', 'yaml', 'sql'],
      },
    }),
};

module.exports = config;
