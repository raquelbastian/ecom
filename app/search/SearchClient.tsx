"use client";

import { useState, useEffect, useRef, FormEvent } from 'react';
import { useSearchParams } from 'next/navigation';
import ProductGrid from '@/components/ProductGrid';
import Link from 'next/link';

interface Product {
  product_id: string;
  product_name: string;
  category: string;
  actual_price: string;
  discounted_price: string;
  rating: string;
  rating_count: string;
  about_product: string;
  img_link: string;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

const ASSISTANT_TOGGLE_KEY = 'aiAssistantEnabled';

export default function SearchClient() {
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [assistantEnabled, setAssistantEnabled] = useState(false);
  const [chat, setChat] = useState<ChatMessage[]>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatInput, setChatInput] = useState('');
  const [chatAvailable, setChatAvailable] = useState(true);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const searchParams = useSearchParams();
  const query = searchParams.get('query');

  // Restore the toggle after mount (localStorage is client-only)
  useEffect(() => {
    setAssistantEnabled(localStorage.getItem(ASSISTANT_TOGGLE_KEY) === 'true');
  }, []);

  const toggleAssistant = () => {
    setAssistantEnabled((prev) => {
      const next = !prev;
      localStorage.setItem(ASSISTANT_TOGGLE_KEY, String(next));
      return next;
    });
  };

  const sendToAssistant = async (history: ChatMessage[]) => {
    setChatLoading(true);
    try {
      const res = await fetch('/api/assistant', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: history }),
      });
      if (res.ok) {
        const data = await res.json();
        if (data.answer) {
          setChat([...history, { role: 'assistant', content: data.answer }]);
        } else if (history.length === 1) {
          // First turn returned nothing — hide the chat entirely
          setChatAvailable(false);
        }
      }
    } catch {
      if (history.length === 1) setChatAvailable(false);
    } finally {
      setChatLoading(false);
    }
  };

  useEffect(() => {
    if (!query) {
      setProducts([]);
      setLoading(false);
      return;
    }

    const fetchProducts = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`/api/products?search=${encodeURIComponent(query)}&vectorSearch=true`);
        if (!res.ok) {
          throw new Error('Failed to fetch search results');
        }
        const data = await res.json();
        setProducts(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setLoading(false);
      }
    };

    fetchProducts();
  }, [query]);

  // Start (or restart) the assistant conversation when the toggle is on
  useEffect(() => {
    if (!query || !assistantEnabled) return;
    const initialChat: ChatMessage[] = [{ role: 'user', content: query }];
    setChat(initialChat);
    setChatAvailable(true);
    sendToAssistant(initialChat);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, assistantEnabled]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [chat, chatLoading]);

  const handleChatSubmit = (e: FormEvent) => {
    e.preventDefault();
    const text = chatInput.trim();
    if (!text || chatLoading) return;
    const history: ChatMessage[] = [...chat, { role: 'user', content: text }];
    setChat(history);
    setChatInput('');
    sendToAssistant(history);
  };

  return (
    <div className="container mx-auto p-4">
      <Link href="/" className="text-blue-500 hover:underline mb-4 block">&larr; Back to Home</Link>

      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-bold">Search Results for "{query}"</h1>

        <button
          type="button"
          onClick={toggleAssistant}
          aria-pressed={assistantEnabled}
          className="flex items-center gap-2 rounded-full border border-indigo-200 bg-white px-3 py-1.5 text-sm shadow-sm transition hover:border-indigo-300"
        >
          <span className={assistantEnabled ? 'font-medium text-indigo-600' : 'text-gray-500'}>
            ✨ AI Shopping Assistant
          </span>
          <span
            className={`relative inline-flex h-5 w-9 items-center rounded-full transition ${
              assistantEnabled ? 'bg-indigo-600' : 'bg-gray-300'
            }`}
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition ${
                assistantEnabled ? 'translate-x-4' : 'translate-x-0.5'
              }`}
            />
          </span>
        </button>
      </div>

      {assistantEnabled && chatAvailable && chat.length > 0 && (
        <div className="mb-6 rounded-lg border border-indigo-200 bg-indigo-50 p-4">
          <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-indigo-500">
            ✨ AI Shopping Assistant
          </p>

          <div className="max-h-80 space-y-3 overflow-y-auto">
            {chat.map((m, i) =>
              m.role === 'assistant' ? (
                <div key={i} className="max-w-[85%] rounded-lg rounded-tl-none bg-white p-3 text-sm text-gray-800 shadow-sm">
                  <p className="whitespace-pre-line">{m.content}</p>
                </div>
              ) : (
                <div key={i} className="ml-auto max-w-[85%] rounded-lg rounded-tr-none bg-indigo-600 p-3 text-sm text-white">
                  <p className="whitespace-pre-line">{m.content}</p>
                </div>
              )
            )}
            {chatLoading && (
              <div className="max-w-[85%] rounded-lg rounded-tl-none bg-white p-3 text-sm shadow-sm">
                <p className="animate-pulse text-indigo-400">Typing...</p>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          <form onSubmit={handleChatSubmit} className="mt-3 flex gap-2">
            <input
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder="Ask a follow-up — e.g. 'is there a cheaper one?'"
              className="flex-1 rounded-md border border-indigo-200 bg-white px-3 py-2 text-sm text-gray-800 placeholder-gray-400 focus:border-indigo-400 focus:outline-none"
            />
            <button
              type="submit"
              disabled={chatLoading || !chatInput.trim()}
              className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Send
            </button>
          </form>
        </div>
      )}

      {loading && <p>Loading...</p>}
      {error && <p className="text-red-500">Error: {error}</p>}
      {!loading && !error && (
        <>
          {products.length > 0 ? (
            <ProductGrid products={products} />
          ) : (
            <p>No products found.</p>
          )}
        </>
      )}
    </div>
  );
}
