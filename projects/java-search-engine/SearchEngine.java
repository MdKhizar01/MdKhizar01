import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;
import java.util.stream.*;

/** Reconstructed plain-text/TREC parser, indexer, and TF-IDF cosine search. */
public class SearchEngine {
    static class Index implements Serializable {
        private static final long serialVersionUID = 1L;
        TreeMap<String, TreeMap<String,Integer>> documents = new TreeMap<>();
        TreeMap<String, TreeMap<String,Integer>> inverted = new TreeMap<>();
    }
    static TreeMap<String,Integer> tokenize(String text) {
        TreeMap<String,Integer> frequencies = new TreeMap<>();
        Matcher matcher = Pattern.compile("[a-z0-9]+").matcher(text.toLowerCase(Locale.ROOT));
        while (matcher.find()) {
            String term = matcher.group();
            // Simple normalization, deliberately not presented as Porter stemming.
            if (term.length() > 5 && term.endsWith("ing")) term = term.substring(0,term.length()-3);
            else if (term.length() > 4 && term.endsWith("ed")) term = term.substring(0,term.length()-2);
            else if (term.length() > 3 && term.endsWith("s")) term = term.substring(0,term.length()-1);
            frequencies.merge(term,1,Integer::sum);
        }
        return frequencies;
    }
    static void add(Index index, String id, String text) {
        if (id.isBlank() || index.documents.containsKey(id)) throw new IllegalArgumentException("Missing or duplicate docID: " + id);
        TreeMap<String,Integer> terms = tokenize(text);
        index.documents.put(id,terms);
        terms.forEach((term, count) -> index.inverted.computeIfAbsent(term,t -> new TreeMap<>()).put(id,count));
    }
    static Index build(Path folder) throws IOException {
        Index index = new Index();
        List<Path> files;
        try (Stream<Path> stream=Files.walk(folder)) {
            files = stream.filter(Files::isRegularFile).filter(p -> p.toString().matches("(?i).+\\.(txt|sgm|sgml|trec)$")).sorted().collect(Collectors.toList());
        }
        for (Path file: files) {
            String text = Files.readString(file,StandardCharsets.UTF_8);
            Matcher doc = Pattern.compile("(?is)<DOC>(.*?)</DOC>").matcher(text);
            boolean trec = false;
            while (doc.find()) {
                trec = true;
                String block = doc.group(1);
                Matcher id = Pattern.compile("(?is)<DOCNO>(.*?)</DOCNO>").matcher(block);
                if (!id.find()) throw new IllegalArgumentException("TREC document missing DOCNO in " + file);
                String docId=id.group(1).trim();
                String body=block.replaceAll("(?is)<DOCNO>.*?</DOCNO>"," ").replaceAll("<[^>]+>"," ");
                add(index,docId,body);
            }
            if (!trec) add(index,folder.relativize(file).toString(),text.replaceAll("<[^>]+>"," "));
        }
        if (index.documents.isEmpty()) throw new IllegalArgumentException("No supported documents found");
        return index;
    }
    static void save(Index index, Path output) throws IOException {
        Files.createDirectories(output);
        try (ObjectOutputStream out = new ObjectOutputStream(Files.newOutputStream(output.resolve("index.bin")))) {out.writeObject(index);}
        Files.write(output.resolve("vocabulary.txt"), index.inverted.keySet(),StandardCharsets.UTF_8);
        Files.write(output.resolve("docIDs.txt"),index.documents.keySet(),StandardCharsets.UTF_8);
        ArrayList<String> forward=new ArrayList<>(), inverted=new ArrayList<>();
        index.documents.forEach((id, terms) -> terms.forEach((term, count) -> forward.add(id+"\t"+term+"\t"+count)));
        index.inverted.forEach((term, docs) -> docs.forEach((id, count) -> inverted.add(term+"\t"+id+"\t"+count)));
        Files.write(output.resolve("forward_index.txt"),forward,StandardCharsets.UTF_8);
        Files.write(output.resolve("inverted_index.txt"),inverted,StandardCharsets.UTF_8);
        Files.write(output.resolve("postings.txt"),inverted,StandardCharsets.UTF_8);
    }
    static Index load(Path folder) throws IOException, ClassNotFoundException {
        try (ObjectInputStream input=new ObjectInputStream(Files.newInputStream(folder.resolve("index.bin")))) {
            input.setObjectInputFilter(ObjectInputFilter.Config.createFilter("maxdepth=12;maxrefs=10000000;java.base/*;SearchEngine$Index;!*"));
            return (Index)input.readObject();
        }
    }
    static Map<String,Double> vector(Index index, Map<String,Integer> terms) {
        Map<String,Double> result=new HashMap<>();
        terms.forEach((term, count) -> {
            if(index.inverted.containsKey(term)) {
                double idf=Math.log((index.documents.size()+1.0)/(index.inverted.get(term).size()+1.0))+1;
                result.put(term,(1+Math.log(count))*idf);
            }
        });
        double norm=Math.sqrt(result.values().stream().mapToDouble(v -> v*v).sum());
        if(norm>0)result.replaceAll((term,weight)->weight/norm);
        return result;
    }
    static List<Map.Entry<String,Double>> search(Index index, String query) {
        Map<String,Double> q=vector(index,tokenize(query));
        Map<String,Double> scores=new HashMap<>();
        index.documents.forEach((id, terms) -> {
            Map<String,Double> d=vector(index,terms);
            double score=q.entrySet().stream().mapToDouble(e->e.getValue()*d.getOrDefault(e.getKey(),0.0)).sum();
            if(score>0)scores.put(id,score);
        });
        return scores.entrySet().stream().sorted(Map.Entry.<String,Double>comparingByValue().reversed().thenComparing(Map.Entry.comparingByKey())).limit(10).collect(Collectors.toList());
    }
    public static void main(String[] args) throws Exception {
        if(args.length<3) {System.err.println("index <documents-folder> <output-folder> OR search <index-folder> <query>");System.exit(2);}
        if(args[0].equals("index")) {
            Index index=build(Path.of(args[1]));save(index,Path.of(args[2]));
            System.out.println("Indexed "+index.documents.size()+" documents, "+index.inverted.size()+" terms");
        } else if(args[0].equals("search")) {
            List<Map.Entry<String,Double>> results=search(load(Path.of(args[1])),String.join(" ",Arrays.copyOfRange(args,2,args.length)));
            for(Map.Entry<String,Double> result:results)System.out.printf(Locale.ROOT,"%s\t%.6f%n",result.getKey(),result.getValue());
            if(results.isEmpty())System.out.println("No matching documents");
        } else throw new IllegalArgumentException("Unknown command: "+args[0]);
    }
}
