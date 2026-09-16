function parseLine(line) {
  const cells = [];
  let current = '';
  let quoted = false;
  for (const ch of line) {
    if (ch === '"') {
      quoted = !quoted;
    } else if (ch === ',' && !quoted) {
      cells.push(current);
      current = '';
    } else {
      current += ch;
    }
  }
  cells.push(current);
  return cells;
}

function toRecords(lines) {
  if (lines.length === 0) {
    return [];
  }
  const [header, ...rest] = lines;
  const columns = parseLine(header);
  return rest.map((line) => {
    const cells = parseLine(line);
    const record = {};
    columns.forEach((name, i) => {
      record[name] = cells[i] ?? '';
    });
    return record;
  });
}

function groupBy(records, key) {
  return records.reduce((acc, record) => {
    const value = record[key];
    (acc[value] ??= []).push(record);
    return acc;
  }, {});
}

function main() {
  const csv = ['name,role', 'ana,dev', 'ben,ops', 'cara,dev'];
  const grouped = groupBy(toRecords(csv), 'role');
  for (const [role, people] of Object.entries(grouped)) {
    console.log(role, people.map((p) => p.name).join(', '));
  }
}

main();
