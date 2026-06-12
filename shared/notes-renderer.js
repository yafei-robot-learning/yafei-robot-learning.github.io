function escapeHtml(value) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function stripHtmlComments(markdown) {
  return markdown.replace(/<!--[\s\S]*?-->/g, '');
}

function renderInlineText(text) {
  return escapeHtml(text).replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
}

const pythonKeywords = new Set([
  'and',
  'as',
  'assert',
  'async',
  'await',
  'break',
  'class',
  'continue',
  'def',
  'del',
  'elif',
  'else',
  'except',
  'False',
  'finally',
  'for',
  'from',
  'global',
  'if',
  'import',
  'in',
  'is',
  'lambda',
  'None',
  'nonlocal',
  'not',
  'or',
  'pass',
  'raise',
  'return',
  'True',
  'try',
  'while',
  'with',
  'yield',
]);

const pythonBuiltins = new Set([
  'abs',
  'bool',
  'breakpoint',
  'dict',
  'enumerate',
  'float',
  'int',
  'len',
  'list',
  'max',
  'min',
  'np',
  'print',
  'range',
  'self',
  'set',
  'str',
  'sum',
  'tuple',
  'zeros',
]);

function renderPythonCode(code) {
  let html = '';
  let index = 0;

  const appendToken = (className, value) => {
    html += `<span class="${className}">${escapeHtml(value)}</span>`;
  };

  while (index < code.length) {
    const rest = code.slice(index);
    const char = code[index];
    const tripleQuote = rest.startsWith('"""') ? '"""' : rest.startsWith("'''") ? "'''" : null;

    if (char === '#') {
      const endIndex = code.indexOf('\n', index);
      const nextIndex = endIndex === -1 ? code.length : endIndex;
      appendToken('syntax-comment', code.slice(index, nextIndex));
      index = nextIndex;
      continue;
    }

    if (tripleQuote) {
      const endIndex = code.indexOf(tripleQuote, index + 3);
      const nextIndex = endIndex === -1 ? code.length : endIndex + 3;
      appendToken('syntax-string', code.slice(index, nextIndex));
      index = nextIndex;
      continue;
    }

    if (char === '"' || char === "'") {
      let nextIndex = index + 1;
      while (nextIndex < code.length) {
        if (code[nextIndex] === '\\') {
          nextIndex += 2;
          continue;
        }

        if (code[nextIndex] === char) {
          nextIndex += 1;
          break;
        }

        if (code[nextIndex] === '\n') {
          break;
        }

        nextIndex += 1;
      }

      appendToken('syntax-string', code.slice(index, nextIndex));
      index = nextIndex;
      continue;
    }

    const numberMatch = rest.match(/^\b\d+(?:\.\d+)?\b/);
    if (numberMatch) {
      appendToken('syntax-number', numberMatch[0]);
      index += numberMatch[0].length;
      continue;
    }

    const identifierMatch = rest.match(/^[A-Za-z_][A-Za-z0-9_]*/);
    if (identifierMatch) {
      const value = identifierMatch[0];
      if (pythonKeywords.has(value)) {
        appendToken('syntax-keyword', value);
      } else if (pythonBuiltins.has(value)) {
        appendToken('syntax-builtin', value);
      } else {
        html += escapeHtml(value);
      }
      index += value.length;
      continue;
    }

    html += escapeHtml(char);
    index += 1;
  }

  return html;
}

function renderCodeBlock(block) {
  const language = sanitizeLabel(block.language || '').toLowerCase();
  const languageClass = language ? ` language-${escapeHtml(language)}` : '';
  const code = language === 'python' || language === 'py' ? renderPythonCode(block.text) : escapeHtml(block.text);

  return `<pre class="code-block"><code class="syntax-code${languageClass}">${code}</code></pre>`;
}

function buildBlocks(markdown) {
  const lines = stripHtmlComments(markdown).replace(/\r\n/g, '\n').split('\n');
  const blocks = [];
  let paragraph = [];
  let mathLines = [];
  let algorithmLines = [];
  let itemizeLines = [];
  let figureLines = [];
  let codeLines = [];
  let inMath = false;
  let inAlgorithm = false;
  let inItemize = false;
  let inFigure = false;
  let inCode = false;
  let codeLanguage = '';
  let itemizeDepth = 0;

  const flushParagraph = () => {
    if (paragraph.length === 0) {
      return;
    }

    blocks.push({ type: 'paragraph', text: paragraph.join(' ') });
    paragraph = [];
  };

  const flushMath = () => {
    if (mathLines.length === 0) {
      return;
    }

    blocks.push({ type: 'math', text: mathLines.join('\n') });
    mathLines = [];
  };

  const flushAlgorithm = () => {
    if (algorithmLines.length === 0) {
      return;
    }

    blocks.push({ type: 'algorithm', text: algorithmLines.join('\n') });
    algorithmLines = [];
  };

  const flushItemize = () => {
    if (itemizeLines.length === 0) {
      return;
    }

    blocks.push({ type: 'itemize', text: itemizeLines.join('\n') });
    itemizeLines = [];
  };

  const flushFigure = () => {
    if (figureLines.length === 0) {
      return;
    }

    blocks.push({ type: 'figure', text: figureLines.join('\n') });
    figureLines = [];
  };

  const flushCode = () => {
    blocks.push({ type: 'code', text: codeLines.join('\n'), language: codeLanguage });
    codeLines = [];
    codeLanguage = '';
  };

  const appendParagraphText = (text) => {
    const trimmedText = text.trim();

    if (trimmedText) {
      paragraph.push(trimmedText);
    }
  };

  const processMathDelimiters = (line) => {
    let rest = line;

    if (rest === '' && inMath) {
      mathLines.push(rest);
      return;
    }

    while (rest.length > 0 || inMath) {
      if (inMath) {
        const endIndex = rest.indexOf('$$');

        if (endIndex === -1) {
          mathLines.push(rest);
          return;
        }

        const mathText = rest.slice(0, endIndex);
        if (mathText.trim()) {
          mathLines.push(mathText);
        }

        flushMath();
        inMath = false;
        rest = rest.slice(endIndex + 2);
        continue;
      }

      const startIndex = rest.indexOf('$$');

      if (startIndex === -1) {
        appendParagraphText(rest);
        return;
      }

      appendParagraphText(rest.slice(0, startIndex));
      flushParagraph();
      rest = rest.slice(startIndex + 2);

      const endIndex = rest.indexOf('$$');

      if (endIndex === -1) {
        if (rest.trim()) {
          mathLines.push(rest);
        }

        inMath = true;
        return;
      }

      const mathText = rest.slice(0, endIndex);
      if (mathText.trim()) {
        mathLines.push(mathText);
      }

      flushMath();
      rest = rest.slice(endIndex + 2);
    }
  };

  for (const line of lines) {
    const trimmed = line.trim();

    if (trimmed.startsWith('%')) {
      continue;
    }

    if (inCode) {
      if (trimmed.startsWith('```')) {
        flushCode();
        inCode = false;
        continue;
      }

      codeLines.push(line);
      continue;
    }

    if (trimmed.startsWith('```')) {
      flushParagraph();
      inCode = true;
      codeLanguage = trimmed.slice(3).trim();
      codeLines = [];
      continue;
    }

    if (trimmed.startsWith('\\begin{algorithm}')) {
      flushParagraph();
      inAlgorithm = true;
      algorithmLines.push(line);
      continue;
    }

    if (inAlgorithm) {
      algorithmLines.push(line);

      if (trimmed === '\\end{algorithm}') {
        flushAlgorithm();
        inAlgorithm = false;
      }

      continue;
    }

    if (trimmed.startsWith('\\begin{figure}')) {
      flushParagraph();
      inFigure = true;
      figureLines.push(line);
      continue;
    }

    if (inFigure) {
      figureLines.push(line);

      if (trimmed === '\\end{figure}') {
        flushFigure();
        inFigure = false;
      }

      continue;
    }

    if (inItemize) {
      itemizeLines.push(line);

      if (trimmed === '\\begin{itemize}') {
        itemizeDepth += 1;
      }

      if (trimmed === '\\end{itemize}') {
        itemizeDepth -= 1;

        if (itemizeDepth === 0) {
          flushItemize();
          inItemize = false;
        }
      }

      continue;
    }

    if (trimmed === '\\begin{itemize}') {
      flushParagraph();
      inItemize = true;
      itemizeDepth = 1;
      itemizeLines.push(line);
      continue;
    }

    if (inMath || line.includes('$$')) {
      processMathDelimiters(line);
      continue;
    }

    if (trimmed === '') {
      flushParagraph();
      continue;
    }

    if (trimmed === '<div align="center">' || trimmed === '</div>') {
      flushParagraph();
      continue;
    }

    if (line.startsWith('# ')) {
      flushParagraph();
      blocks.push({ type: 'heading', level: 1, text: line.slice(2).trim() });
      continue;
    }

    if (line.startsWith('## ')) {
      flushParagraph();
      blocks.push({ type: 'heading', level: 2, text: line.slice(3).trim() });
      continue;
    }

    if (line.startsWith('### ')) {
      flushParagraph();
      blocks.push({ type: 'heading', level: 3, text: line.slice(4).trim() });
      continue;
    }

    if (trimmed.startsWith('<img ') || trimmed.startsWith('![')) {
      flushParagraph();
      blocks.push({ type: 'image', text: trimmed });
      continue;
    }

    paragraph.push(line.trim());
  }

  flushParagraph();
  flushAlgorithm();
  flushItemize();
  flushFigure();
  if (inCode || codeLines.length > 0) {
    flushCode();
  }

  return blocks;
}

const notesConfig = window.NOTES_CONFIG || {
  title: 'Notes',
  documentTitle: 'Notes',
  notes: [],
};

const notes = notesConfig.notes || [];

function sanitizeLabel(label) {
  return label.trim().replace(/[^A-Za-z0-9_-]/g, '-');
}

function extractEquationMeta(text, number) {
  const labelMatch = text.match(/\\label\{([^}]+)\}/);
  const label = labelMatch ? labelMatch[1].trim() : null;
  const cleanedText = text.replace(/\s*\\label\{[^}]+\}\s*/g, '\n').trim();

  return {
    id: label ? `eq-${sanitizeLabel(label)}` : `equation-${number}`,
    label,
    number,
    text: cleanedText,
  };
}

function formatAlgorithmLine(line) {
  return line
    .replace(/\\While\{\\text\{([^}]+)\}\}/g, 'while $1')
    .replace(/\\While\{([^}]+)\}/g, 'while $1')
    .replace(/\\\\\s*$/g, '')
    .trim();
}

function parseAlgorithm(text) {
  const captionMatch = text.match(/\\caption\{([^}]+)\}/);
  const title = captionMatch ? captionMatch[1].trim() : 'Algorithm';
  const lines = text.split('\n');
  const steps = [];
  let indent = 0;

  for (const rawLine of lines) {
    const trimmed = rawLine.trim();

    if (
      trimmed === '' ||
      trimmed.startsWith('\\begin{algorithm}') ||
      trimmed === '\\end{algorithm}' ||
      trimmed.startsWith('\\caption') ||
      trimmed.startsWith('\\label')
    ) {
      continue;
    }

    if (trimmed === '}') {
      indent = Math.max(0, indent - 1);
      continue;
    }

    const line = formatAlgorithmLine(trimmed);

    if (!line || line === '{') {
      continue;
    }

    steps.push({ text: line, indent });

    if (line.startsWith('while ') || trimmed.endsWith('{')) {
      indent += 1;
    }
  }

  return { title, steps };
}

function parseItemizeLines(lines, startIndex = 0) {
  const items = [];
  let index = startIndex;
  let currentItem = null;

  while (index < lines.length) {
    const trimmed = lines[index].trim();

    if (trimmed === '\\begin{itemize}') {
      if (currentItem) {
        const result = parseItemizeLines(lines, index + 1);
        currentItem.children.push(result.items);
        index = result.index;
        continue;
      }

      index += 1;
      continue;
    }

    if (trimmed === '\\end{itemize}') {
      return { items, index: index + 1 };
    }

    if (trimmed.startsWith('\\item')) {
      currentItem = {
        text: trimmed.replace(/^\\item\s*/, '').trim(),
        children: [],
      };
      items.push(currentItem);
      index += 1;
      continue;
    }

    if (currentItem && trimmed) {
      currentItem.text = `${currentItem.text} ${trimmed}`.trim();
    }

    index += 1;
  }

  return { items, index };
}

function renderItemizeItems(items) {
  return `<ul>${items
    .map((item) => {
      const children = item.children.map((childItems) => renderItemizeItems(childItems)).join('');
      return `<li>${renderInlineText(item.text)}${children}</li>`;
    })
    .join('')}</ul>`;
}

function renderItemize(text) {
  const lines = text.split('\n');
  const result = parseItemizeLines(lines);

  return renderItemizeItems(result.items);
}

function parseFigure(text) {
  const imageMatch = text.match(/\\includegraphics(?:\[([^\]]+)\])?\{([^}]+)\}/);
  const captionMatch = text.match(/\\caption\{([^}]*)\}/);
  const options = imageMatch ? imageMatch[1] || '' : '';
  const widthMatch = options.match(/width=([0-9.]+)\\linewidth/);
  const width = widthMatch ? `${Number(widthMatch[1]) * 100}%` : null;

  return {
    src: imageMatch ? imageMatch[2].trim() : '',
    caption: captionMatch ? captionMatch[1].trim() : '',
    width,
  };
}

function resolveNoteAssetPath(src, noteFileName) {
  if (/^(https?:|data:|\/)/.test(src)) {
    return src;
  }

  const noteDir = noteFileName.includes('/') ? noteFileName.slice(0, noteFileName.lastIndexOf('/') + 1) : '';
  return new URL(`${noteDir}${src}`, window.location.href).href;
}

function renderFigure(text, noteFileName) {
  const figure = parseFigure(text);

  if (!figure.src) {
    return '';
  }

  const widthStyle = figure.width ? ` style="--figure-width: ${escapeHtml(figure.width)}"` : '';
  const caption = figure.caption ? `<figcaption>${escapeHtml(figure.caption)}</figcaption>` : '';

  return `
    <figure class="note-figure"${widthStyle}>
      <img src="${escapeHtml(resolveNoteAssetPath(figure.src, noteFileName))}" alt="${escapeHtml(figure.caption || 'Note figure')}" loading="lazy" />
      ${caption}
    </figure>
  `;
}

function parseImage(text) {
  const markdownMatch = text.match(/^!\[([^\]]*)\]\((\S+)(?:\s+"([^"]+)")?\)$/);

  if (markdownMatch) {
    const widthMatch = (markdownMatch[3] || '').match(/width=([^\s]+)/);

    return {
      src: markdownMatch[2].trim(),
      alt: markdownMatch[1].trim(),
      width: widthMatch ? widthMatch[1].trim() : '',
    };
  }

  const srcMatch = text.match(/\bsrc="([^"]+)"/);
  const altMatch = text.match(/\balt="([^"]*)"/);
  const widthMatch = text.match(/\bwidth="([^"]+)"/);

  return {
    src: srcMatch ? srcMatch[1].trim() : '',
    alt: altMatch ? altMatch[1].trim() : '',
    width: widthMatch ? widthMatch[1].trim() : '',
  };
}

function renderImage(text, noteFileName) {
  const image = parseImage(text);

  if (!image.src) {
    return '';
  }

  const widthStyle = image.width ? ` style="--figure-width: ${escapeHtml(image.width)}"` : '';

  return `
    <figure class="note-figure"${widthStyle}>
      <img src="${escapeHtml(resolveNoteAssetPath(image.src, noteFileName))}" alt="${escapeHtml(image.alt || 'Note figure')}" loading="lazy" />
    </figure>
  `;
}

function replaceEquationRefs(root, equationMap) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const textNodes = [];

  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (node.parentElement && node.parentElement.closest('.math-block')) {
      continue;
    }

    if (/\\(eqref|ref)\{[^}]+\}/.test(node.nodeValue)) {
      textNodes.push(node);
    }
  }

  for (const node of textNodes) {
    const parent = node.parentNode;
    if (!parent) {
      continue;
    }

    const fragment = document.createDocumentFragment();
    const pattern = /\\(eqref|ref)\{([^}]+)\}/g;
    let lastIndex = 0;
    let match;

    while ((match = pattern.exec(node.nodeValue)) !== null) {
      if (match.index > lastIndex) {
        fragment.appendChild(document.createTextNode(node.nodeValue.slice(lastIndex, match.index)));
      }

      const command = match[1];
      const label = match[2].trim();
      const equation = equationMap.get(label);
      const anchor = document.createElement('a');
      anchor.className = 'equation-ref';
      anchor.href = equation ? `#${equation.id}` : '#';
      anchor.textContent = equation ? (command === 'eqref' ? `(${equation.number})` : `${equation.number}`) : '??';
      fragment.appendChild(anchor);
      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < node.nodeValue.length) {
      fragment.appendChild(document.createTextNode(node.nodeValue.slice(lastIndex)));
    }

    parent.replaceChild(fragment, node);
  }
}

async function renderMarkdownNote(fileName, content) {
  try {
    const response = await fetch(fileName, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`Failed to load note: ${response.status}`);
    }

    const markdown = await response.text();
    const blocks = buildBlocks(markdown);
    let equationNumber = 1;
    const equationMap = new Map();

    const renderedBlocks = blocks.map((block) => {
      if (block.type !== 'math') {
        return block;
      }

      const equation = extractEquationMeta(block.text, equationNumber);
      equationNumber += 1;

      if (equation.label) {
        equationMap.set(equation.label, equation);
      }

      return {
        ...block,
        equation,
      };
    });

    content.innerHTML = renderedBlocks
      .map((block) => {
        if (block.type === 'heading') {
          return `<h${block.level}>${escapeHtml(block.text)}</h${block.level}>`;
        }

        if (block.type === 'math') {
          return `
            <div class="math-block" id="${escapeHtml(block.equation.id)}">
              <div class="math-block-body"></div>
              <a class="equation-number" href="#${escapeHtml(block.equation.id)}">(${block.equation.number})</a>
            </div>
          `;
        }

        if (block.type === 'algorithm') {
          const algorithm = parseAlgorithm(block.text);
          const steps = algorithm.steps
            .map((step) => `
              <div class="algorithm-line" style="--indent: ${step.indent}">
                ${escapeHtml(step.text)}
              </div>
            `)
            .join('');

          return `
            <figure class="algorithm-block">
              <figcaption>${escapeHtml(algorithm.title)}</figcaption>
              <div class="algorithm-body">${steps}</div>
            </figure>
          `;
        }

        if (block.type === 'itemize') {
          return renderItemize(block.text);
        }

        if (block.type === 'figure') {
          return renderFigure(block.text, fileName);
        }

        if (block.type === 'image') {
          return renderImage(block.text, fileName);
        }

        if (block.type === 'code') {
          return renderCodeBlock(block);
        }

        return `<p>${renderInlineText(block.text)}</p>`;
      })
      .join('');

    const mathBlocks = content.querySelectorAll('.math-block-body');
    let mathIndex = 0;

    for (const block of renderedBlocks) {
      if (block.type !== 'math') {
        continue;
      }

      const target = mathBlocks[mathIndex];
      if (target && window.katex) {
        katex.render(block.equation.text, target, {
          displayMode: true,
          throwOnError: false,
          macros: {
            '\\vertbar': '\\vert',
          },
        });
      } else if (target) {
        target.textContent = block.equation.text;
      }

      mathIndex += 1;
    }

    replaceEquationRefs(content, equationMap);

    if (window.renderMathInElement) {
      renderMathInElement(content, {
        delimiters: [
          { left: '$', right: '$', display: false },
        ],
        throwOnError: false,
        macros: {
          '\\vertbar': '\\vert',
        },
      });
    }
  } catch (error) {
    content.innerHTML = `<p class="loading">Unable to load ${escapeHtml(fileName)}. ${escapeHtml(error.message)}</p>`;
  }
}

function loadContentsPage() {
  const contentsNav = document.querySelector('.contents');
  const sectionTitle = document.querySelector('[data-section-title]');

  if (sectionTitle) {
    sectionTitle.textContent = notesConfig.title;
  }

  document.title = `${notesConfig.documentTitle || notesConfig.title} | Contents`;

  if (!contentsNav) {
    return;
  }

  contentsNav.innerHTML = notes
    .map((note) => `
      <a class="contents-item" href="${escapeHtml(note.path)}">
        <span class="contents-title">${escapeHtml(note.title)}</span>
      </a>
    `)
    .join('');
}

async function loadNotePage() {
  const content = document.getElementById('content');
  const noteParam = new URLSearchParams(window.location.search).get('note') || 'LR.md';
  const note = notes.find((item) => item.file === noteParam || item.file.endsWith(`/${noteParam}`)) || notes[0];

  if (note) {
    document.title = `${notesConfig.documentTitle || notesConfig.title} | ${note.title}`;
  }

  if (content) {
    content.innerHTML = '<div class="loading">Loading note…</div>';
    await renderMarkdownNote(note.file, content);
  }
}

function loadPage() {
  if (document.body.dataset.page === 'note') {
    loadNotePage();
    return;
  }

  loadContentsPage();
}

document.addEventListener('DOMContentLoaded', loadPage);
