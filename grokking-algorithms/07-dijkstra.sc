//> using scala "3"
//> using dep "org.scalacheck::scalacheck::1.18.0"
import scala.collection.immutable.HashMap

type Graph[A] = HashMap[A, List[Edge[A]]]

final case class Edge[A](from: A, to: A, weight: Int)

trait Showable[A]:
  extension (a: A) def show: String

given Showable[List[Edge[Int]]] with
  extension (nodes: List[Edge[Int]])
    def show: String =
      nodes.foldLeft("") {
        case ("", n) =>
          s"${n.from} -> ${n.to} (${n.weight})"
        case (str, n) =>
          str + s" -> ${n.from} -> ${n.to} (${n.weight})"
      }

val graph = HashMap(
  "book" -> List(
    Edge("book", "lp", 5),
    Edge("book", "poster", 0)
  ),
  "lp" -> List(
    Edge("lp", "guitar", 15),
    Edge("lp", "drum", 20)
  ),
  "poster" -> List(
    Edge("poster", "guitar", 30),
    Edge("poster", "drum", 35)
  ),
  "guitar" -> List(
    Edge("guitar", "piano", 20)
  ),
  "drum" -> List(
    Edge("drum", "piano", 10)
  )
)

def costTable(graph: Graph[String], start: String): HashMap[String, Int] =
  graph.foldLeft(HashMap[String, Int].empty) { case (acc, ()) =>

  }

def cheapestNode(graph: Graph[Int], start: String): String =
  graph(start).reduce { case (e1, e2) =>
    if e1.weight < e2.weight then e1 else e2
  }.to

println(cheapestNode("book"))

def djikstra(
    graph: Graph[String],
    from: String,
    to: String
): List[Edge[String]] =
  List.empty
