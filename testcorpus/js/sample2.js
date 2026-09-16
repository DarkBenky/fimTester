const path = require('path');

class TaskList {
  constructor() {
    this.tasks = [];
  }

  add(title, priority = 0) {
    this.tasks.push({ title, priority, done: false });
    return this;
  }

  complete(title) {
    const task = this.tasks.find((t) => t.title === title);
    if (!task) {
      throw new Error(`unknown task: ${title}`);
    }
    task.done = true;
    return task;
  }

  pending() {
    return this.tasks.filter((t) => !t.done);
  }

  byPriority() {
    return [...this.tasks].sort((a, b) => b.priority - a.priority);
  }
}

function summarize(tasks) {
  const done = tasks.filter((t) => t.done).length;
  return `${done}/${tasks.length} done`;
}

function main() {
  const list = new TaskList();
  list.add('write tests', 2).add('fix bug', 5).add('update docs', 1);
  list.complete('fix bug');
  for (const task of list.byPriority()) {
    console.log(`${task.done ? '[x]' : '[ ]'} ${task.title}`);
  }
  console.log(summarize(list.tasks));
  console.log('file:', path.basename(__filename));
}

main();
