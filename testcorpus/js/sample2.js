const fs = require('fs');

function sum(values) {
  let total = 0;
  for (const v of values) {
    total += v;
  }
  return total;
}

function filterMin(values, min) {
  const out = [];
  for (const v of values) {
    if (v > min) {
      out.push(v);
    }
  }
  return out;
}

function greet(name) {
  return `hello ${name}`;
}

function main() {
  const name = process.argv[2] || 'world';
  console.log(greet(name));
  const nums = [1, 2, 3, 4, 5, 6, 7, 8];
  const big = filterMin(nums, 4);
  console.log(sum(big));
}

main();
