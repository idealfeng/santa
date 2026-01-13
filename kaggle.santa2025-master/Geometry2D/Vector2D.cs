using System;

namespace kaggle.santa2025.Geometry2D
{
    public readonly struct Vector2D(double x, double y) : IEquatable<Vector2D>
    {
        public double X { get; init; } = x;
        public double Y { get; init; } = y;

        public Vector2D() : this(0.0, 0.0) { }

        public static Vector2D Zero { get; } = new(0.0, 0.0);
        public static Vector2D UnitX { get; } = new(1.0, 0.0);
        public static Vector2D UnitY { get; } = new(0.0, 1.0);

        public static Vector2D operator +(Vector2D a, Vector2D b) => new(a.X + b.X, a.Y + b.Y);
        public static Vector2D operator -(Vector2D a, Vector2D b) => new(a.X - b.X, a.Y - b.Y);
        public static Vector2D operator -(Vector2D a) => new(-a.X, -a.Y);
        public static Vector2D operator *(Vector2D v, double s) => new(v.X * s, v.Y * s);
        public static Vector2D operator *(double s, Vector2D v) => v * s;
        public static Vector2D operator /(Vector2D v, double s) => new(v.X / s, v.Y / s);

        public static Vector2D Min(Vector2D a, Vector2D b) => new(Math.Min(a.X, b.X), Math.Min(a.Y, b.Y));
        public static Vector2D Max(Vector2D a, Vector2D b) => new(Math.Max(a.X, b.X), Math.Max(a.Y, b.Y));

        public static double Dot(Vector2D a, Vector2D b) => a.X * b.X + a.Y * b.Y;

        public readonly double Length() => Math.Sqrt(X * X + Y * Y);
        public readonly double LengthSquared() => X * X + Y * Y;

        public readonly Vector2D Normalized()
        {
            double len = Length();
            return len > 1e-9 ? this / len : Zero;
        }

        public readonly Vector2D Translated(Vector2D delta) => this + delta;
        public readonly Vector2D Translated(double deltaX, double deltaY) => new(X + deltaX, Y + deltaY);

        public readonly Vector2D Rotated(double angleRadians)
        {
            double c = Math.Cos(angleRadians);
            double s = Math.Sin(angleRadians);
            return new Vector2D(X * c - Y * s, X * s + Y * c);
        }

        public readonly Vector2D Scaled(double scaleX, double scaleY) => new(X * scaleX, Y * scaleY);
        public readonly Vector2D Scaled(double uniformScale) => Scaled(uniformScale, uniformScale);

        public void Deconstruct(out double x, out double y) => (x, y) = (X, Y);

        public override string ToString() => $"({X:F2}, {Y:F2})";

        public bool Equals(Vector2D other) => X == other.X && Y == other.Y;
        public override bool Equals(object? obj) => obj is Vector2D v && Equals(v);
        public override int GetHashCode() => HashCode.Combine(X, Y);
        public static bool operator ==(Vector2D left, Vector2D right) => left.Equals(right);
        public static bool operator !=(Vector2D left, Vector2D right) => !left.Equals(right);
    }
}
