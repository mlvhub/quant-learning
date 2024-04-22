//> using scala "3"
//> using dep "org.scalacheck::scalacheck::1.18.0"

final case class Node[A](value: A, neighbors: List[Node[A]])

trait Showable[A]:
  extension (a: A) def show: String

given Showable[List[Node[Int]]] with
  extension (nodes: List[Node[Int]])
    def show: String =
      nodes.foldLeft("") {
        case ("", n) =>
          s"${n.value}"
        case (str, n) =>
          str + s" -> ${n.value}"
      }

// depth-first approach, inefficient due to `flatMap` and `sortBy`, there's also risk of infinite recursion
// def shortestPath[A](
//     node1: Node[A],
//     node2: Node[A]
// ): Option[List[Node[A]]] =
//   def find(
//       neighbors: List[Node[A]],
//       previous: List[Node[A]]
//   ): Option[List[Node[A]]] =
//     neighbors
//       .flatMap(n =>
//         if n == node2 then Some(previous :+ n)
//         else find(n.neighbors, previous :+ n)
//       )
//       .sortBy(_.length)
//       .headOption

//   find(node1.neighbors, List.empty)

def shortestPath[A](start: Node[A], goal: Node[A]): Option[List[Node[A]]] =
  import scala.collection.mutable

  val visited = mutable.HashMap[A, Unit]()
  val queue = mutable.Queue[(Node[A], List[Node[A]])]()
  queue.enqueueAll(start.neighbors.map(n => (n, List(n))))

  while (queue.nonEmpty)
    queue.dequeue() match
      case (currentNode, path) if (currentNode == goal) =>
        return Some(path)
      case (currentNode, path) =>
        visited.update(currentNode.value, ())
        for (
          neighbor <- currentNode.neighbors if !visited.contains(neighbor.value)
        )
          queue.enqueue((neighbor, path :+ neighbor))
  None

// def connected[A](node1: Node[A], node2: Node[A]): Boolean =
//   def find(neighbors: List[Node[A]]): Option[Node[A]] =
//     neighbors
//       .flatMap(n => if n == node2 then Some(node2) else find(n.neighbors))
//       .headOption

//   find(node1.neighbors).isDefined

def connected[A](node1: Node[A], node2: Node[A]): Boolean =
  shortestPath[A](node1, node2).isDefined

val n6 = Node(6, List.empty)
val n5 = Node(5, List(n6))
val n4 = Node(4, List(n5))
val n3 = Node(3, List(n4))
val n2 = Node(2, List(n6))
val n1 = Node(1, List(n2, n3))

val tests = List(
  (n1, List(n2, n3, n4, n5, n6), true),
  (n2, List(n6), true),
  (n2, List(n1, n3, n4, n5), false),
  (n3, List(n1, n2), false),
  (n3, List(n4, n5, n6), true),
  (n4, List(n1, n2, n3), false),
  (n4, List(n5, n6), true),
  (n5, List(n1, n2, n3, n4), false),
  (n5, List(n6), true),
  (n6, List(n1, n2, n3, n4, n5), false)
)
tests.foreach { case (node1, nodes2, expected) =>
  nodes2.foreach(n2 =>
    val actual = connected(node1, n2) == expected
    assert(
      actual,
      s"n1: ${node1.value} - n2: ${n2.value}: expected ${expected} but got ${actual}"
    )
  )
}
println("All tests ran successfully!")

println(shortestPath(n1, n6).get.show)
println(shortestPath(n1, n3).get.show)
