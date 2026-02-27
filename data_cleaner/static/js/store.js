import { ref } from "vue";

export const cardCount = ref(null);

export async function fetchCount() {
  if (cardCount.value !== null) return;
  const r = await fetch("/api/cards/count");
  const d = await r.json();
  cardCount.value = d.count;
}
