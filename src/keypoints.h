
#ifndef _KEY_POINTS_H_
#define _KEY_POINTS_H_

// STL
#include <iostream>
#include <ostream>
#include <string>

// Cereal
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>

class KeyPoint {
 public:
  KeyPoint() {}
  KeyPoint(float x, float y, float size, float angle, float responce, int octave)
      : x_(x), y_(y), size_(size), angle_(angle), responce_(responce), octave_(octave) {}

  bool operator==(const KeyPoint& rhs) const {
    return x_ == rhs.x_ && y_ == rhs.y_ && size_ == rhs.size_ && angle_ == rhs.angle_ &&
           responce_ == rhs.responce_ && octave_ == rhs.octave_;
  }

  bool operator!=(const KeyPoint& rhs) const { return !this->operator==(rhs); }

  friend std::ostream& operator<<(std::ostream& output, const KeyPoint& kp) {
    output << kp.x_ << ", " << kp.y_ << ", " << kp.size_ << ", " << kp.angle_ << ", "
           << kp.responce_ << ", " << kp.octave_;
    return output;
  }

 public:
  float x_;
  float y_;
  float size_;
  float angle_;
  float responce_;
  int octave_;

  friend cereal::access;
  template <class Archive>
  void serialize(Archive& ar) {
    ar(x_, y_, size_, angle_, responce_, octave_);
  }
};

#endif